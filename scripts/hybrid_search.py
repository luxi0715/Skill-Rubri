"""
hybrid_search.py
----------------
BM25 + 向量混合检索，对比三种模式结果：
  bm25   -- 纯关键词检索
  vector -- 纯向量语义检索
  hybrid -- 加权融合（默认 alpha=0.7 向量 + 0.3 BM25）

运行方式：
  python hybrid_search.py --query "contrastive learning cold start" --mode hybrid --top 3
  python hybrid_search.py --query "NeuralUCB routing" --mode all --top 3
"""

import argparse
import json
import os
import re

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
INDEX_API = os.path.join(DATA_DIR, "paper_index_api.faiss")

API_MODEL = "text-embedding-3-small"

# ── P4 Fix：绝对质量阈值 ──────────────────────────────────
# FAISS 内积（余弦）分数：< 0.10 代表几乎不相关，整批丢弃而非相对归一
FAISS_MIN_QUALITY = 0.10

# BM25 原始分阈值：低于此值说明没有有效关键词匹配，整批丢弃
# 标定方法：对已知相关 query（标题自检索）观察 raw max 分布，取 10th percentile
# 164 篇语料实测：标题自检索 raw max 约在 2.0~8.0，OOD 查询约在 0.01~0.3
BM25_MIN_RAW_SCORE = 1.0

# ── FAISS 索引：模块加载时读入一次，常驻内存 ─────────────
_faiss_index = faiss.read_index(INDEX_API) if os.path.exists(INDEX_API) else None


# ── 加载数据 ──────────────────────────────────────────────
def load_eval_results() -> list:
    results_file = os.path.join(DATA_DIR, "eval_results.json")
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 文本处理 ──────────────────────────────────────────────
def tokenize(text: str) -> list:
    """英文分词：正则提取小写字母和数字"""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_corpus(records: list) -> list:
    """
    每篇论文拼成一段文字，用于 BM25。

    P5 Fix：优先使用原始摘要（abstract），保留专有名词（GraphSAGE、BERT 等）。
    GPT 改写后专有名词会被泛化（"graph sampling method"），导致关键词匹配失效。
    """
    corpus = []
    for r in records:
        parts = [
            r.get("paper", ""),           # 标题（最重要，专有名词密度高）
            r.get("abstract", ""),        # P5 Fix：原始摘要，专有名词完整
            r.get("research_question", ""),
            r.get("methodology", ""),
            r.get("datasets", ""),
            r.get("results", ""),
            r.get("comment", ""),
        ]
        corpus.append(" ".join(p for p in parts if p))
    return corpus


# ── BM25 检索 ─────────────────────────────────────────────
def search_bm25(query: str, records: list, top_k: int) -> dict:
    """
    返回 {rowid: 归一化分数} 字典。

    P4 Fix（完整版）：
    - max_s <= 0：完全无 token 命中，返回空
    - max_s < BM25_MIN_RAW_SCORE：有偶然 token 重叠（如 "flow" 在 "data flow" 中），
      但得分极低说明无真实语义相关，返回空
    - 否则归一化返回
    """
    corpus           = build_corpus(records)
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25   = BM25Okapi(tokenized_corpus)
    scores = np.array(bm25.get_scores(tokenize(query)), dtype="float32")
    max_s  = float(scores.max())
    if max_s < BM25_MIN_RAW_SCORE:
        return {}   # P4 Fix：原始分低于阈值，即使有微弱匹配也视为无相关结果
    scores = scores / max_s
    return {r["rowid"]: float(scores[i]) for i, r in enumerate(records)}


# ── 向量检索 ──────────────────────────────────────────────
def encode_api(query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp   = client.embeddings.create(model=API_MODEL, input=[query])
    vec    = np.array([resp.data[0].embedding], dtype="float32")
    norm   = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.clip(norm, 1e-10, None)


def search_vector(query_vec: np.ndarray, top_k: int) -> dict:
    """
    返回 {rowid: 归一化分数} 字典，rowid 直接来自 FAISS IndexIDMap2。

    P4 Fix：若最高原始余弦分 < FAISS_MIN_QUALITY（默认 0.10），
    说明语料库中根本没有语义相关论文，返回空字典而非相对归一化的虚假高分。
    """
    if _faiss_index is None:
        return {}   # Bug Fix：索引未加载时安全返回空，不崩溃
    index       = _faiss_index
    n           = index.ntotal
    scores, ids = index.search(query_vec, n)
    result = {}
    for rid, score in zip(ids[0], scores[0]):
        if rid >= 0:
            result[int(rid)] = max(float(score), 0)
    max_s = max(result.values()) if result else 0.0
    # P4 Fix：绝对质量门控 — 最佳匹配余弦 < 0.10 整体视为无效
    if max_s < FAISS_MIN_QUALITY:
        return {}
    result = {k: v / max_s for k, v in result.items()}
    return result


# ── 混合融合 ──────────────────────────────────────────────
def hybrid_scores(bm25_scores: dict, vec_scores: dict,
                  alpha: float = 0.7) -> dict:
    """按 rowid 对齐，alpha * vector + (1-alpha) * bm25"""
    all_ids = set(bm25_scores) | set(vec_scores)
    return {
        rid: alpha * vec_scores.get(rid, 0) + (1 - alpha) * bm25_scores.get(rid, 0)
        for rid in all_ids
    }


# ── P3 Fix：Cross-Encoder 精排 ────────────────────────────
# Bi-Encoder（FAISS）是召回模型，训练目标是快速缩小候选集。
# Cross-Encoder 对 (query, doc) 对逐一打分，精度高 3-5 倍，但速度慢。
# 架构：Bi-Encoder 召回 top-50 → Cross-Encoder 精排 → 取 top-k 返回用户。
#
# 模型：cross-encoder/ms-marco-MiniLM-L-6-v2
#   - ~80MB，CPU 可跑，推理 50 条 ~300ms
#   - 首次使用自动下载（需联网）
#
_cross_encoder = None   # 延迟加载，不影响模块启动时间

def _get_cross_encoder():
    """延迟加载 CrossEncoder，确保导入失败时不崩溃整个模块。"""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _cross_encoder
    except Exception:
        return None


def rerank_with_cross_encoder(query: str, candidates: list[dict],
                               text_field: str = "text") -> list[dict]:
    """
    P3 Fix：用 Cross-Encoder 对候选列表精排。

    参数：
        query      — 用户查询字符串
        candidates — list of dict，每个 dict 需包含 text_field 指定的文本字段
        text_field — 用于打分的文本字段名（默认 "text"）

    返回：
        按 cross_encoder_score 降序排列的 candidates，每条增加 cross_encoder_score 字段。
        若 CrossEncoder 不可用（未安装 / 网络问题），原样返回 candidates。
    """
    if not candidates:
        return candidates

    ce = _get_cross_encoder()
    if ce is None:
        return candidates   # 降级：保持 Bi-Encoder 顺序

    pairs  = [(query, c.get(text_field, "")) for c in candidates]
    scores = ce.predict(pairs)

    for c, s in zip(candidates, scores):
        c["cross_encoder_score"] = float(s)

    return sorted(candidates, key=lambda x: -x.get("cross_encoder_score", 0))


# ── 格式化输出 ────────────────────────────────────────────
def print_results(results: list, label: str):
    print(f"\n{'─'*60}")
    print(f"  [{label}]")
    print(f"{'─'*60}")
    for r in results:
        print(f"  #{r['rank']}  {r['paper']}")
        print(f"      score: {r['score']:.4f}  |  quality: {r['quality_score']}  |  rubric: {r['rubric']:.2f}")
        comment = r['comment']
        if len(comment) > 100:
            comment = comment[:100] + "..."
        print(f"      {comment}")
        print()


def top_k_results(scores: dict, top_k: int, label: str) -> list:
    """按分数取 top-k，用 rowid 直接查 SQLite"""
    # 这三行代码的作用是把当前脚本所在的目录强行加到 Python 模块搜索路径的最前面，
    # 确保不管你从哪里运行这个脚本，都能正确找到同目录下的 db 模块并导入 get_paper_by_rowid。
    import sys
    # 当前正在运行的这个 Python 文件（比如 hybrid_search.py）的完整路径
    # .insert(0, ...)：把上面那个文件夹插到列表第 0 位（最前面）。
    sys.path.insert(0, os.path.dirname(__file__))
    from db import get_paper_by_rowid

    # sorted(scores)
    # 默认会把 scores 这个字典的 key（也就是 rid） 拿出来排序。
    # key=scores.__getitem__
    # 关键点：告诉 sorted 不要按 rid 排序，而是按每个 rid 对应的分数（value） 来排序。
    # reverse=True
    # 直接从高到低排序（最高分排在最前面）。
    # 不是“先从小到大排，再反转”。
    # Python 内部是一次性按降序排好的（效率更高）。
    # [:top_k]
    # 排好序后，只取前 top_k 个 rid（比如 top_k=3 就只留前3名）。
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    results = []
    for rank, rid in enumerate(sorted_ids, 1):
        r = get_paper_by_rowid(rid)
        if not r:
            continue
        results.append({
            "rank":          rank,
            "paper":         r["paper"],
            "score":         round(scores[rid], 4),
            "quality_score": r.get("paper_quality_score", 0),
            "rubric":        r.get("skill_rubric_total", 0),
            "comment":       r.get("comment_cn") or r.get("comment", ""),
        })
    return results


# ── 相似论文推荐 ──────────────────────────────────────────
def recommend_similar(paper_name: str, records: list, top_k: int = 5) -> list:
    """
    基于内容的推荐：用 rowid 从 IndexIDMap2 重建向量，找最近邻。
    不再依赖 paper_meta.json。
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from db import get_paper_by_rowid

    index = _faiss_index
    n     = index.ntotal

    # 从 records 找该论文的 rowid
    paper_rowid = None
    for r in records:
        if r.get("paper") == paper_name or paper_name in r.get("paper", ""):
            paper_rowid = r.get("rowid")
            break

    if paper_rowid is None:
        return []

    # IndexIDMap2 支持按 ID 重建向量
    vec = np.zeros((1, index.d), dtype="float32")
    index.reconstruct(paper_rowid, vec[0])

    # 搜索最相似的 top_k+1（含自身）
    scores, ids = index.search(vec, min(top_k + 2, n))

    results = []
    for rid, score in zip(ids[0], scores[0]):
        if rid < 0 or rid == paper_rowid:
            continue
        r = get_paper_by_rowid(int(rid)) or {}
        pname = r.get("paper", "")
        if not pname:
            continue
        results.append({
            "paper":               pname,
            "score":               round(float(score), 4),
            "title_cn":            r.get("title_cn", ""),
            "paper_quality_score": r.get("paper_quality_score", 0),
            "comment":             r.get("comment_cn") or r.get("comment", ""),
            "domain":              r.get("domain", ""),
            "year":                r.get("year", ""),
        })
        if len(results) >= top_k:
            break

    return results


# ── 主流程 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode",  choices=["bm25", "vector", "hybrid", "all"],
                        default="hybrid")
    parser.add_argument("--top",   type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="向量权重（hybrid模式），默认0.7")
    args = parser.parse_args()

    print(f"\n查询：{args.query}  |  模式：{args.mode}  |  top-{args.top}")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from db import get_all_papers
    records = get_all_papers()   # 从 SQLite 读，含 rowid

    # BM25 分数（直接传 records，内部自己 build_corpus）
    bm25_scores = search_bm25(args.query, records, args.top)

    # 向量分数
    query_vec  = encode_api(args.query)
    vec_scores = search_vector(query_vec, args.top)

    modes = ["bm25", "vector", "hybrid"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "bm25":
            scores = bm25_scores
            label  = "BM25 (keyword)"
        elif mode == "vector":
            scores = vec_scores
            label  = "Vector (semantic)"
        else:
            scores = hybrid_scores(bm25_scores, vec_scores, args.alpha)
            label  = f"Hybrid (alpha={args.alpha})"

        results = top_k_results(scores, args.top, label)
        print_results(results, label)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # 右键 Run 时跑测试
        sys.path.insert(0, os.path.dirname(__file__))
        from db import get_all_papers
        records = get_all_papers()   # 从 SQLite 读，含 rowid

        test_queries = [
            "contrastive learning recommendation",
            "NeuralUCB reward routing",
        ]
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"查询：{query}")
            bm25_scores = search_bm25(query, records, top_k=3)
            query_vec   = encode_api(query)
            vec_scores  = search_vector(query_vec, top_k=3)
            hy_scores   = hybrid_scores(bm25_scores, vec_scores, alpha=0.7)

            for scores, label in [
                (bm25_scores, "BM25 (keyword)"),
                (vec_scores,  "Vector (semantic)"),
                (hy_scores,   "Hybrid alpha=0.7"),
            ]:
                results = top_k_results(scores, top_k=3, label=label)
                print_results(results, label)
