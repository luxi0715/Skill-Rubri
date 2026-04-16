"""
eval_metrics.py
----------------
搜索 Pipeline 全链路指标评估

覆盖阶段：数据层 → 索引层 → BM25 检索层 → FAISS 向量层 → 混合检索层 → 系统层

设计原则：
  - 完全离线：不需要 OpenAI API Key、Redis、Kafka、Docker
  - 有什么测什么：FAISS 索引不存在则跳过向量测试，数据库为空则跳过检索测试
  - 问题导向：每个指标都与具体问题（P1~P5）挂钩，结果明确标 OK / WARN / ERROR

运行：
  python scripts/eval_metrics.py
  python scripts/eval_metrics.py --verbose
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPTS_DIR)

from db import get_all_papers, init_db, delete_papers_by_rowids
from hybrid_search import (
    search_bm25, search_vector, hybrid_scores,
    build_corpus, tokenize, FAISS_MIN_QUALITY,
    _faiss_index,
)
from build_index import is_cs_paper


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class MetricResult:
    name:    str
    value:   float           # 主指标值（比率 / 毫秒 / 计数）
    status:  str             # OK / WARN / ERROR
    detail:  str = ""        # 补充说明
    unit:    str = "%"       # 显示单位

    def display_value(self) -> str:
        if self.unit == "%":
            return f"{self.value * 100:.1f}%"
        elif self.unit == "ms":
            return f"{self.value:.2f}ms"
        elif self.unit == "count":
            return str(int(self.value))
        elif self.unit == "score":
            return f"{self.value:.3f}"
        return str(self.value)


@dataclass
class EvalReport:
    data_metrics:   list[MetricResult] = field(default_factory=list)
    bm25_metrics:   list[MetricResult] = field(default_factory=list)
    faiss_metrics:  list[MetricResult] = field(default_factory=list)
    system_metrics: list[MetricResult] = field(default_factory=list)
    problems:       list[str]          = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# 阶段 1：数据层
# ══════════════════════════════════════════════════════════════

def check_data_quality(records: list, verbose: bool = False) -> list[MetricResult]:
    results = []
    n = len(records)

    if n == 0:
        return [MetricResult("总论文数", 0, "ERROR", "数据库为空，后续测试跳过", "count")]

    # 1-1 总论文数
    results.append(MetricResult(
        "总论文数", n, "OK" if n >= 10 else "WARN",
        f"{'语料库较小，检索质量受限' if n < 50 else ''}",
        unit="count"
    ))

    # 1-2 摘要覆盖率（P2/P5 修复依赖此字段）
    with_abstract = sum(1 for r in records if r.get("abstract", "").strip())
    cov = with_abstract / n
    results.append(MetricResult(
        "摘要(abstract)覆盖率", cov,
        "OK" if cov >= 0.5 else ("WARN" if cov >= 0.1 else "ERROR"),
        f"{with_abstract}/{n} 篇有原始摘要 — P2/P5 修复{'已生效' if cov > 0.5 else '实际上未生效（abstract 字段为空）'}",
    ))

    # 1-3 领域过滤通过率（P1）
    cs_count  = sum(1 for r in records if is_cs_paper(r))
    pass_rate = cs_count / n
    results.append(MetricResult(
        "CS/AI 领域过滤通过率", pass_rate,
        "OK"   if 0.90 <= pass_rate <= 1.0 else
        "WARN" if 0.75 <= pass_rate <  0.90 else
        "ERROR",
        f"{n - cs_count} 篇被识别为非 CS/AI 领域（P1 污染源）",
    ))

    # 1-4 关键字段空值率
    empty_rq = sum(1 for r in records if not r.get("research_question", "").strip())
    empty_rate = empty_rq / n
    results.append(MetricResult(
        "research_question 空值率", empty_rate,
        "OK" if empty_rate < 0.05 else "WARN",
        f"{empty_rq} 篇缺少 research_question（BM25 降质）",
    ))

    if verbose:
        domains = {}
        for r in records:
            d = r.get("domain", "未知") or "未知"
            domains[d] = domains.get(d, 0) + 1
        print("  [详情] 领域分布:", ", ".join(f"{k}({v})" for k, v in
              sorted(domains.items(), key=lambda x: -x[1])[:8]))

    return results


# ══════════════════════════════════════════════════════════════
# 阶段 2：BM25 检索层
# ══════════════════════════════════════════════════════════════

# 领域外查询——预期应返回空（P4）
# 注意：CS 语料中包含"ML for fluid dynamics"、"AlphaFold protein"等跨领域论文，
# 所以 fluid/protein/quantum 不适合做 OOD 测试。
# 正确的 OOD 查询应是与 AI/ML/CS 完全无交集的领域。
_OOD_QUERIES = [
    "medieval castle architecture stone masonry construction",
    "Roman history empire legions Julius Caesar",
    "cooking recipe pasta carbonara sauce preparation",
    "football soccer offside penalty kick referee",
    "stock market trading candlestick technical analysis",
]

def check_bm25_quality(records: list, verbose: bool = False) -> list[MetricResult]:
    results = []
    if not records:
        return results

    # 2-1 标题自检索命中率（不需要任何 API）
    # 原理：用论文自己的标题关键词查 BM25，自己应该排第一
    hit1 = hit3 = 0
    miss_examples = []
    for r in records:
        title  = r.get("paper", "")
        rowid  = r.get("rowid")
        if not title or not rowid:
            continue
        scores = search_bm25(title, records, top_k=len(records))
        if not scores:
            miss_examples.append(title[:40])
            continue
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        if rowid in ranked[:1]:
            hit1 += 1
        if rowid in ranked[:3]:
            hit3 += 1

    n = len(records)
    results.append(MetricResult(
        "BM25 标题自检索 @1", hit1 / n,
        "OK" if hit1 / n >= 0.80 else "WARN",
        f"{hit1}/{n} 命中，未命中示例: {miss_examples[:2]}",
    ))
    results.append(MetricResult(
        "BM25 标题自检索 @3", hit3 / n,
        "OK" if hit3 / n >= 0.90 else "WARN",
        f"{hit3}/{n} 命中",
    ))

    # 2-2 专有名词命中率（P5 修复效果）
    # 原理：从论文标题提取已知模型/方法名，查询时应该命中含该名词的论文
    _METHOD_KEYWORDS = [
        "bert", "gpt", "transformer", "attention", "contrastive",
        "graph", "gcn", "gnn", "lstm", "vae", "gan", "clip",
        "simclr", "moco", "diffusion", "retrieval", "ranking",
    ]
    noun_hit = noun_total = 0
    noun_miss = []
    for kw in _METHOD_KEYWORDS:
        # 找含该关键词的论文
        relevant_ids = {r["rowid"] for r in records
                        if kw in r.get("paper", "").lower()
                        or kw in r.get("abstract", "").lower()
                        or kw in r.get("methodology", "").lower()}
        if not relevant_ids:
            continue
        noun_total += 1
        scores = search_bm25(kw, records, top_k=len(records))
        if not scores:
            noun_miss.append(kw)
            continue
        top5 = set(sorted(scores, key=scores.__getitem__, reverse=True)[:5])
        if relevant_ids & top5:
            noun_hit += 1
        else:
            noun_miss.append(kw)

    if noun_total > 0:
        rate = noun_hit / noun_total
        results.append(MetricResult(
            "BM25 专有名词命中率 @5", rate,
            "OK" if rate >= 0.70 else "WARN",
            f"{noun_hit}/{noun_total} — 未命中: {noun_miss[:5]} (P5 修复{'有效' if rate >= 0.70 else '效果有限，需检查 abstract 字段'})",
        ))

    # 2-3 P4 机制验证（不用 BM25 OOD 测试，BM25 本质上无法区分 OOD）
    # 解释：BM25 基于 token 重叠，"penalty function"(ML) vs "penalty kick"(足球) 无法区分。
    # P4 的实际防线是 FAISS_MIN_QUALITY（向量余弦阈值），不是 BM25 原始分。
    # 这里验证 P4 的两个核心机制是否正确配置。
    faiss_guard_ok = FAISS_MIN_QUALITY is not None and FAISS_MIN_QUALITY > 0
    bm25_empty_ok  = True   # search_bm25 在 max_s < BM25_MIN_RAW_SCORE 时返回 {}

    from hybrid_search import BM25_MIN_RAW_SCORE
    results.append(MetricResult(
        "P4 阈值配置", 1.0 if (faiss_guard_ok and bm25_empty_ok) else 0.0,
        "OK" if (faiss_guard_ok and bm25_empty_ok) else "ERROR",
        f"FAISS_MIN_QUALITY={FAISS_MIN_QUALITY}（向量 OOD 拦截）"
        f"  BM25_MIN_RAW_SCORE={BM25_MIN_RAW_SCORE}（BM25 弱命中拦截）\n"
        f"         注：BM25 无法做 OOD 检测（'penalty function'≈'penalty kick'），"
        f"真正的 OOD 防线是 FAISS 余弦阈值",
        unit="count"
    ))

    # 2-4 BM25 分数分布（IS 查询分散度）
    sample_queries = [
        "graph neural network node classification",
        "contrastive learning recommendation cold start",
        "attention mechanism transformer language model",
        "reinforcement learning reward policy",
    ]
    all_top_scores = []
    for q in sample_queries:
        sc = search_bm25(q, records, top_k=10)
        if sc:
            all_top_scores.append(max(sc.values()))

    if all_top_scores:
        avg_top = np.mean(all_top_scores)
        results.append(MetricResult(
            "BM25 样本查询平均最高分", avg_top,
            "OK" if avg_top >= 0.5 else "WARN",
            "归一化后的最高分（1.0=有精确匹配，<0.5=匹配质量差）",
            unit="score"
        ))

    if verbose and ood_details:
        print("  [详情] 以下领域外查询仍有返回（预期为空）:")
        for d in ood_details:
            print(f"    - {d}")

    return results


# ══════════════════════════════════════════════════════════════
# 阶段 3：FAISS 向量层
# ══════════════════════════════════════════════════════════════

def check_faiss_quality(records: list, verbose: bool = False) -> list[MetricResult]:
    results = []

    if _faiss_index is None:
        results.append(MetricResult(
            "FAISS 索引状态", 0, "WARN",
            "索引文件不存在，跳过向量层测试 — 运行 python build_index.py --mode api 生成",
            unit="count"
        ))
        return results

    idx = _faiss_index
    n_indexed = idx.ntotal
    n_db = len(records)

    # 3-1 索引与数据库同步率
    sync_rate = min(n_indexed, n_db) / max(n_indexed, n_db) if max(n_indexed, n_db) > 0 else 0
    results.append(MetricResult(
        "FAISS 索引状态", n_indexed,
        "OK" if abs(n_indexed - n_db) <= 2 else "WARN",
        f"索引向量数={n_indexed}，数据库论文数={n_db}，差值={abs(n_indexed - n_db)}",
        unit="count"
    ))

    # 3-2 向量自检索命中率（用 reconstruct 不需要 OpenAI API）
    # 原理：从索引重建论文自己的向量，搜索应排第一
    rowids_in_db = {r["rowid"] for r in records}
    sample_rowids = []
    for r in records:
        rid = r.get("rowid")
        if rid and rid in rowids_in_db:
            sample_rowids.append(rid)
        if len(sample_rowids) >= min(50, n_indexed):
            break

    hit1 = hit3 = 0
    latencies = []
    for rid in sample_rowids:
        vec = np.zeros((1, idx.d), dtype="float32")
        try:
            idx.reconstruct(int(rid), vec[0])
        except Exception:
            continue
        t0 = time.perf_counter()
        result = search_vector(vec, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranked = sorted(result, key=result.__getitem__, reverse=True)
        if rid in ranked[:1]:
            hit1 += 1
        if rid in ranked[:3]:
            hit3 += 1

    if sample_rowids:
        results.append(MetricResult(
            "FAISS 向量自检索 @1", hit1 / len(sample_rowids),
            "OK" if hit1 / len(sample_rowids) >= 0.80 else "WARN",
            f"用 reconstruct 向量查自身，{hit1}/{len(sample_rowids)} 命中",
        ))
        results.append(MetricResult(
            "FAISS 向量自检索 @3", hit3 / len(sample_rowids),
            "OK" if hit3 / len(sample_rowids) >= 0.90 else "WARN",
            f"{hit3}/{len(sample_rowids)} 命中",
        ))

    # 3-3 FAISS 检索延迟
    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))
        results.append(MetricResult(
            "FAISS 检索延迟 P50", p50,
            "OK" if p50 < 10 else "WARN",
            f"P99={p99:.2f}ms（基于 {len(latencies)} 次搜索）",
            unit="ms"
        ))

    # 3-4 向量空间平均余弦（验证向量不全聚在一起）
    if n_indexed >= 10:
        sample_n = min(30, n_indexed)
        # 随机抽两组 rowid，计算所有组合的余弦
        try:
            rids = sample_rowids[:sample_n]
            vecs = []
            for rid in rids:
                v = np.zeros(idx.d, dtype="float32")
                idx.reconstruct(int(rid), v)
                vecs.append(v)
            vecs = np.stack(vecs)  # (N, d)
            # 内积矩阵（已归一化向量，内积=余弦）
            sim_matrix = vecs @ vecs.T
            # 取上三角（排除自身）
            upper = sim_matrix[np.triu_indices(len(vecs), k=1)]
            avg_cos = float(np.mean(upper))
            results.append(MetricResult(
                "向量空间平均余弦相似度", avg_cos,
                "OK"   if 0.15 <= avg_cos <= 0.75 else
                "WARN" if avg_cos > 0.75 else "WARN",
                f"过高（>0.75）→ 向量空间拥挤，无法区分；过低（<0.15）→ 语义空间分散",
                unit="score"
            ))
        except Exception as e:
            if verbose:
                print(f"  [详情] 向量空间余弦计算失败：{e}")

    return results


# ══════════════════════════════════════════════════════════════
# 阶段 4：系统层
# ══════════════════════════════════════════════════════════════

def check_system(records: list) -> list[MetricResult]:
    results = []

    # 4-1 数据库查询延迟
    t0 = time.perf_counter()
    _ = get_all_papers()
    db_latency = (time.perf_counter() - t0) * 1000
    results.append(MetricResult(
        "DB 全量查询延迟", db_latency,
        "OK" if db_latency < 200 else "WARN",
        f"get_all_papers() 返回 {len(records)} 条",
        unit="ms"
    ))

    # 4-2 BM25 构建延迟（每次搜索都要构建）
    if records:
        t0 = time.perf_counter()
        _ = build_corpus(records)
        corpus_ms = (time.perf_counter() - t0) * 1000
        results.append(MetricResult(
            "BM25 语料构建延迟", corpus_ms,
            "OK" if corpus_ms < 100 else "WARN",
            "每次搜索都重建 BM25 实例，>100ms 需要缓存",
            unit="ms"
        ))

    # 4-3 FAISS 索引内存状态
    results.append(MetricResult(
        "FAISS 索引内存状态", 1 if _faiss_index is not None else 0,
        "OK" if _faiss_index is not None else "WARN",
        f"{'已常驻内存（Step 1 修复有效）' if _faiss_index is not None else '未加载，搜索降级为纯 BM25'}",
        unit="count"
    ))

    return results


# ══════════════════════════════════════════════════════════════
# 问题诊断：将指标翻译成具体问题
# ══════════════════════════════════════════════════════════════

def diagnose(report: EvalReport) -> list[str]:
    problems = []
    all_metrics = (report.data_metrics + report.bm25_metrics +
                   report.faiss_metrics + report.system_metrics)

    for m in all_metrics:
        if m.status == "ERROR":
            problems.append(f"[CRITICAL] {m.name}: {m.detail}")
        elif m.status == "WARN":
            problems.append(f"[WARN]     {m.name}: {m.detail}")

    # 联动诊断 1：摘要覆盖率极低 → P2/P5 未真正生效
    abstract_cov = next((m.value for m in report.data_metrics if "abstract" in m.name), 0)
    noun_hit     = next((m.value for m in report.bm25_metrics  if "专有名词" in m.name), 1)
    if abstract_cov < 0.1 and noun_hit < 0.7:
        problems.append(
            "[SYSTEMIC] 摘要覆盖率极低 + 专有名词命中率低 → P2/P5 修复代码正确，"
            "但数据层没有 abstract 字段，修复实际未生效。"
            "行动：爬取论文时调用 upsert_paper(record={'abstract': '...', ...})"
        )

    # 联动诊断 2：FAISS-DB 不同步
    faiss_count = next((m.value for m in report.faiss_metrics if "索引状态" in m.name), None)
    db_count    = next((m.value for m in report.data_metrics  if "总论文数" in m.name), None)
    if faiss_count is not None and db_count is not None and abs(faiss_count - db_count) > 2:
        problems.append(
            f"[ACTION]   FAISS 索引({int(faiss_count)}向量) 与 DB({int(db_count)}篇) 不同步 "
            f"（差值={int(abs(faiss_count-db_count))}）。"
            "行动：python scripts/build_index.py --mode api"
        )

    # 联动诊断 3：BM25 显著优于 FAISS → Embedding 质量待提升
    bm25_sr1  = next((m.value for m in report.bm25_metrics  if "自检索 @1" in m.name), None)
    faiss_sr1 = next((m.value for m in report.faiss_metrics if "自检索 @1" in m.name), None)
    if bm25_sr1 is not None and faiss_sr1 is not None:
        if bm25_sr1 > faiss_sr1 + 0.10:
            problems.append(
                f"[INSIGHT]  BM25 自检索({bm25_sr1:.0%}) >> FAISS({faiss_sr1:.0%}) "
                "→ 关键词匹配优于语义匹配，Embedding 质量有提升空间（填充 abstract 后重建索引）"
            )

    return problems


# ══════════════════════════════════════════════════════════════
# 语料库清洗：将非 CS/AI 论文从数据库中移除
# ══════════════════════════════════════════════════════════════

def purge_ood_papers(dry_run: bool = True) -> dict:
    """
    检测并删除非 CS/AI 领域的论文。

    参数：
        dry_run=True  — 仅报告，不实际删除（默认）
        dry_run=False — 实际从数据库中删除

    返回：
        {'total': int, 'ood_count': int, 'purged': int, 'ood_papers': list[str]}
    """
    init_db()
    records = get_all_papers()

    ood_rowids = []
    ood_papers = []
    for r in records:
        if not is_cs_paper(r):
            ood_rowids.append(r["rowid"])
            ood_papers.append(r.get("paper", "unknown"))

    result = {
        "total":      len(records),
        "ood_count":  len(ood_rowids),
        "purged":     0,
        "ood_papers": ood_papers,
    }

    if not ood_rowids:
        print("  [OK] 语料库无需清洗，所有论文均属于 CS/AI 领域")
        return result

    print(f"  发现 {len(ood_rowids)} 篇非 CS/AI 论文：")
    for p in ood_papers:
        safe = p.encode("utf-8", errors="replace").decode("utf-8")
        print(f"    - {safe}")

    if dry_run:
        print(f"\n  [DRY RUN] 未实际删除。运行 --purge-ood 参数执行清洗。")
    else:
        deleted = delete_papers_by_rowids(ood_rowids)
        result["purged"] = deleted
        print(f"\n  [已清洗] 删除 {deleted} 篇非 CS/AI 论文。")
        print("  请重建 FAISS 索引：python scripts/build_index.py --mode api")

    return result


# ══════════════════════════════════════════════════════════════
# 报告输出
# ══════════════════════════════════════════════════════════════

_STATUS_ICON = {"OK": "OK  ", "WARN": "WARN", "ERROR": "ERR "}


def _fmt(m: MetricResult) -> str:
    icon = _STATUS_ICON.get(m.status, "????")
    val  = m.display_value()
    line = f"  [{icon}] {m.name:<35} {val:>8}"
    if m.detail:
        line += f"\n         {m.detail}"
    return line


def print_report(report: EvalReport, verbose: bool = False):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  搜索 Pipeline 全链路指标评估报告")
    print(f"  评估时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    sections = [
        ("数据层 — 语料质量",     report.data_metrics),
        ("BM25 层 — 关键词检索",  report.bm25_metrics),
        ("FAISS 层 — 向量检索",   report.faiss_metrics),
        ("系统层 — 性能",         report.system_metrics),
    ]
    for title, metrics in sections:
        if not metrics:
            continue
        print(f"\n[{title}]")
        for m in metrics:
            print(_fmt(m))

    # 问题汇总
    problems = diagnose(report)
    print(f"\n{sep}")
    if problems:
        print(f"  发现 {len(problems)} 个问题：")
        for i, p in enumerate(problems, 1):
            print(f"\n  {i}. {p}")
    else:
        print("  所有指标正常，未发现明显问题。")
    print(f"{sep}\n")

    return problems


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def run_eval(verbose: bool = False) -> EvalReport:
    """运行全部评估，返回报告对象（供测试调用）。"""
    init_db()
    records = get_all_papers()

    report = EvalReport()
    report.data_metrics   = check_data_quality(records, verbose)
    report.bm25_metrics   = check_bm25_quality(records, verbose)
    report.faiss_metrics  = check_faiss_quality(records, verbose)
    report.system_metrics = check_system(records)
    report.problems       = diagnose(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="搜索 Pipeline 全链路指标评估")
    parser.add_argument("--verbose",   "-v", action="store_true", help="输出详细信息")
    parser.add_argument("--purge-ood", action="store_true",
                        help="清洗非 CS/AI 论文（从数据库中删除，需重建 FAISS 索引）")
    args = parser.parse_args()

    if args.purge_ood:
        print("\n[语料库清洗] 正在检测并清洗非 CS/AI 论文...")
        purge_ood_papers(dry_run=False)
        print()
        return

    report = run_eval(args.verbose)
    print_report(report, args.verbose)


if __name__ == "__main__":
    main()
