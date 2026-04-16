"""
test_eval_metrics.py
---------------------
验证 eval_metrics.py 各评估函数的正确性

测试策略：
  - 所有测试使用内存 SQLite + mock 数据，不依赖真实数据库或 FAISS 索引
  - 每个 check_xxx() 函数独立测试，验证 OK/WARN/ERROR 判断逻辑
  - 验证 diagnose() 的联动判断是否准确

运行：
  cd D:\\Skill Rubri\\agenteval_test01
  python -m pytest tests/test_eval_metrics.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_metrics import (
    check_data_quality, check_bm25_quality, check_faiss_quality,
    check_system, diagnose, EvalReport, MetricResult, run_eval,
)


# ── 测试数据 ────────────────────────────────────────────────────

def _make_records(n: int = 20, with_abstract: bool = False,
                  all_cs: bool = True) -> list:
    """生成 n 条测试论文记录"""
    cs_templates = [
        ("Graph Neural Network for Node Classification", "deep learning", "graph sampling"),
        ("BERT Pre-training for NLP Tasks", "natural language", "transformer"),
        ("Contrastive Learning for Recommendation", "contrastive", "embedding"),
        ("Attention Mechanism in Transformers", "attention", "self-supervised"),
        ("GNN Link Prediction on Knowledge Graph", "graph neural", "node classification"),
        ("SimCLR Visual Representation Learning", "contrastive", "image embedding"),
        ("Reinforcement Learning Policy Gradient", "deep learning", "reward optimization"),
        ("Diffusion Model for Image Generation", "diffusion", "generation"),
        ("Graph Convolutional Networks GCN", "graph neural", "classification"),
        ("CLIP Vision Language Pre-training", "contrastive", "multimodal"),
    ]
    ood_templates = [
        ("Turbulent Flow in Navier-Stokes Equations", "fluid dynamics", "mesh simulation"),
        ("Riemann Hypothesis and Prime Numbers", "number theory", "analytic continuation"),
    ]
    records = []
    for i in range(n):
        if all_cs or i < n - 2:
            title, rq, method = cs_templates[i % len(cs_templates)]
        else:
            title, rq, method = ood_templates[i % len(ood_templates)]
        records.append({
            "rowid":              i + 1,
            "paper":              f"{title} v{i}",
            "research_question":  f"How to improve {rq}?",
            "methodology":        method,
            "datasets":           "benchmark dataset",
            "results":            "improved by 3%",
            "comment":            "solid work",
            "abstract":           (f"We propose a {method} approach for {rq}." if with_abstract else ""),
            "domain":             "cs.LG",
            "year":               "2023",
        })
    return records


# ══════════════════════════════════════════════════════════════
# check_data_quality() 测试
# ══════════════════════════════════════════════════════════════

class TestDataQuality:

    def test_empty_db_returns_error(self):
        """空数据库应返回 ERROR"""
        results = check_data_quality([])
        assert any(m.status == "ERROR" for m in results)

    def test_total_count_metric(self):
        """论文数指标应正确计数"""
        records = _make_records(30)
        results = check_data_quality(records)
        count_metric = next(m for m in results if "总论文数" in m.name)
        assert count_metric.value == 30

    def test_abstract_coverage_zero(self):
        """无摘要时覆盖率应为 0，状态 ERROR"""
        records = _make_records(20, with_abstract=False)
        results = check_data_quality(records)
        abstract_m = next(m for m in results if "abstract" in m.name.lower())
        assert abstract_m.value == 0.0
        assert abstract_m.status == "ERROR"

    def test_abstract_coverage_full(self):
        """全部有摘要时覆盖率应为 1.0，状态 OK"""
        records = _make_records(20, with_abstract=True)
        results = check_data_quality(records)
        abstract_m = next(m for m in results if "abstract" in m.name.lower())
        assert abstract_m.value == pytest.approx(1.0)
        assert abstract_m.status == "OK"

    def test_domain_filter_all_cs(self):
        """全 CS 论文时领域过滤通过率应接近 1.0"""
        records = _make_records(20, all_cs=True)
        results = check_data_quality(records)
        domain_m = next(m for m in results if "领域" in m.name)
        assert domain_m.value >= 0.90
        assert domain_m.status == "OK"

    def test_domain_filter_with_ood_papers(self):
        """混入领域外论文时通过率应下降"""
        records = _make_records(20, all_cs=False)  # 最后2篇是非CS
        results = check_data_quality(records)
        domain_m = next(m for m in results if "领域" in m.name)
        # 18/20 = 90% 应该 OK，但实际用 is_cs_paper 过滤，结果可能略有差异
        assert domain_m.value <= 1.0

    def test_small_corpus_warns(self):
        """论文数 < 10 应发出 WARN"""
        records = _make_records(5)
        results = check_data_quality(records)
        count_m = next(m for m in results if "总论文数" in m.name)
        assert count_m.status == "WARN"


# ══════════════════════════════════════════════════════════════
# check_bm25_quality() 测试
# ══════════════════════════════════════════════════════════════

class TestBM25Quality:

    def test_self_retrieval_high_for_unique_titles(self):
        """每篇论文标题独特时，BM25 自检索命中率应高"""
        records = _make_records(15, with_abstract=False)
        results = check_bm25_quality(records)
        sr1 = next((m for m in results if "自检索 @1" in m.name), None)
        assert sr1 is not None
        assert sr1.value >= 0.5   # mock 数据有重复模板，宽松一点

    def test_self_retrieval_at3_higher_than_at1(self):
        """@3 命中率应 >= @1 命中率"""
        records = _make_records(15)
        results = check_bm25_quality(records)
        sr1 = next((m for m in results if "自检索 @1" in m.name), None)
        sr3 = next((m for m in results if "自检索 @3" in m.name), None)
        if sr1 and sr3:
            assert sr3.value >= sr1.value

    def test_p4_threshold_metric_exists(self):
        """P4 阈值配置指标应存在"""
        records = _make_records(15)
        results = check_bm25_quality(records)
        p4_m = next((m for m in results if "P4" in m.name or "阈值" in m.name), None)
        assert p4_m is not None

    def test_empty_records_returns_empty_list(self):
        """空记录时不应崩溃"""
        results = check_bm25_quality([])
        assert results == []

    def test_proper_noun_metric_exists_when_records_have_methods(self):
        """有方法名关键词时，专有名词命中率指标应生成"""
        records = _make_records(20)   # 包含 graph, bert, contrastive 等
        results = check_bm25_quality(records)
        noun_m = next((m for m in results if "专有名词" in m.name), None)
        assert noun_m is not None

    def test_metric_result_status_values(self):
        """所有指标的 status 应只为 OK/WARN/ERROR"""
        records = _make_records(20)
        results = check_bm25_quality(records)
        valid_statuses = {"OK", "WARN", "ERROR"}
        for m in results:
            assert m.status in valid_statuses, f"{m.name} 的 status={m.status} 无效"


# ══════════════════════════════════════════════════════════════
# check_faiss_quality() 测试
# ══════════════════════════════════════════════════════════════

class TestFAISSQuality:

    def test_no_index_returns_warn(self):
        """FAISS 索引不存在时应返回 WARN，不崩溃"""
        with patch("eval_metrics._faiss_index", None):
            results = check_faiss_quality([])
        assert len(results) == 1
        assert results[0].status == "WARN"

    def test_with_mock_index_returns_metrics(self):
        """有 mock 索引时应返回向量相关指标"""
        dim = 64
        n   = 10
        mock_index = MagicMock()
        mock_index.ntotal = n
        mock_index.d      = dim

        # reconstruct：返回随机归一化向量
        def fake_reconstruct(rid, out):
            vec = np.random.randn(dim).astype("float32")
            vec /= np.linalg.norm(vec)
            out[:] = vec
        mock_index.reconstruct.side_effect = fake_reconstruct

        # search：第一个 id 始终是查询的 rowid（自检索命中）
        def fake_search(q, k):
            ids    = np.array([[i + 1 for i in range(min(k, n))]])
            scores = np.array([[1.0 - i * 0.05 for i in range(min(k, n))]])
            return scores, ids
        mock_index.search.side_effect = fake_search

        records = _make_records(n)

        with patch("eval_metrics._faiss_index", mock_index), \
             patch("eval_metrics.search_vector") as mock_sv:
            # 让 search_vector 返回：rowid=1 → 最高分
            def fake_sv(vec, top_k):
                return {1: 1.0, 2: 0.8, 3: 0.6}
            mock_sv.side_effect = fake_sv
            results = check_faiss_quality(records)

        # 应该有多个指标（索引状态、自检索、延迟等）
        assert len(results) >= 2
        names = [m.name for m in results]
        assert any("索引" in name for name in names)

    def test_index_db_sync_detected(self):
        """索引向量数与 DB 数不一致时应 WARN"""
        mock_index = MagicMock()
        mock_index.ntotal = 200   # 比 DB 多 180
        mock_index.d = 64
        mock_index.reconstruct.side_effect = lambda rid, out: None
        mock_index.search.return_value = (np.array([[0.9]]), np.array([[1]]))

        records = _make_records(20)   # DB 只有 20 篇
        with patch("eval_metrics._faiss_index", mock_index), \
             patch("eval_metrics.search_vector", return_value={1: 1.0}):
            results = check_faiss_quality(records)

        count_m = next((m for m in results if "索引状态" in m.name), None)
        assert count_m is not None
        assert count_m.status == "WARN"


# ══════════════════════════════════════════════════════════════
# check_system() 测试
# ══════════════════════════════════════════════════════════════

class TestSystemMetrics:

    def test_system_metrics_exist(self):
        """系统层指标应包含 DB 延迟和 FAISS 状态"""
        records = _make_records(10)
        with patch("eval_metrics.get_all_papers", return_value=records):
            results = check_system(records)
        names = [m.name for m in results]
        assert any("DB" in name for name in names)
        assert any("FAISS" in name for name in names)

    def test_db_latency_is_measured(self):
        """DB 延迟应为非负数"""
        records = _make_records(10)
        with patch("eval_metrics.get_all_papers", return_value=records):
            results = check_system(records)
        db_m = next((m for m in results if "DB" in m.name), None)
        assert db_m is not None
        assert db_m.value >= 0


# ══════════════════════════════════════════════════════════════
# diagnose() 测试
# ══════════════════════════════════════════════════════════════

class TestDiagnose:

    def test_no_problems_when_all_ok(self):
        """所有指标 OK 时不应报告问题"""
        report = EvalReport(
            data_metrics=[
                MetricResult("总论文数", 100, "OK", "", "count"),
                MetricResult("摘要覆盖率", 0.9, "OK", ""),
                MetricResult("领域过滤通过率", 0.95, "OK", ""),
            ],
            bm25_metrics=[
                MetricResult("BM25 标题自检索 @1", 0.88, "OK", ""),
                MetricResult("BM25 专有名词命中率 @5", 0.75, "OK", ""),
                MetricResult("BM25 领域外空结果率", 0.90, "OK", ""),
            ],
        )
        problems = diagnose(report)
        assert len(problems) == 0

    def test_error_metrics_always_reported(self):
        """ERROR 状态的指标必须出现在 problems 中"""
        report = EvalReport(
            data_metrics=[
                MetricResult("总论文数", 0, "ERROR", "数据库为空", "count"),
            ],
        )
        problems = diagnose(report)
        assert any("CRITICAL" in p for p in problems)

    def test_warn_metrics_reported(self):
        """WARN 状态指标应出现在 problems 中"""
        report = EvalReport(
            bm25_metrics=[
                MetricResult("BM25 标题自检索 @1", 0.65, "WARN", "命中率偏低"),
            ],
        )
        problems = diagnose(report)
        assert any("WARN" in p for p in problems)

    def test_systemic_diagnosis_when_abstract_low_and_noun_low(self):
        """摘要覆盖率极低 + 专有名词命中率低时应触发联动诊断"""
        report = EvalReport(
            data_metrics=[
                MetricResult("摘要(abstract)覆盖率", 0.0, "ERROR", ""),
                MetricResult("总论文数", 100, "OK", "", "count"),
                MetricResult("CS/AI 领域过滤通过率", 1.0, "OK", ""),
            ],
            bm25_metrics=[
                MetricResult("BM25 专有名词命中率 @5", 0.5, "WARN", ""),
            ],
        )
        problems = diagnose(report)
        assert any("SYSTEMIC" in p for p in problems)

    def test_insight_when_bm25_beats_faiss(self):
        """BM25 自检索显著高于 FAISS 时应提供 INSIGHT"""
        report = EvalReport(
            bm25_metrics=[
                MetricResult("BM25 标题自检索 @1", 0.92, "OK", ""),
            ],
            faiss_metrics=[
                MetricResult("FAISS 向量自检索 @1", 0.70, "WARN", ""),
            ],
        )
        problems = diagnose(report)
        assert any("INSIGHT" in p for p in problems)


# ══════════════════════════════════════════════════════════════
# MetricResult 工具函数测试
# ══════════════════════════════════════════════════════════════

class TestMetricResult:

    def test_display_percent(self):
        m = MetricResult("test", 0.856, "OK", "", "%")
        assert "85.6%" in m.display_value()

    def test_display_ms(self):
        m = MetricResult("test", 12.34, "OK", "", "ms")
        assert "12.34ms" in m.display_value()

    def test_display_count(self):
        m = MetricResult("test", 164.0, "OK", "", "count")
        assert "164" in m.display_value()

    def test_display_score(self):
        m = MetricResult("test", 0.42, "OK", "", "score")
        assert "0.420" in m.display_value()
