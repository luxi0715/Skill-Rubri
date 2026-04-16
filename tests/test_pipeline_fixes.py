"""
test_pipeline_fixes.py
-----------------------
验证论文搜索 Pipeline 五个问题的修复。

问题映射：
  P1 — 数据清洗：领域过滤，防止流体力学等非 CS 论文污染向量空间
  P2 — Embedding 质量：build_text() 优先用原始摘要而非 GPT 压缩字段
  P3 — 缺精排：Cross-Encoder Rerank 对 Bi-Encoder 召回结果二次打分
  P4 — 缺评估兜底：绝对阈值过滤，归一化分数低于门控时返回空
  P5 — BM25 专有名词丢失：BM25 语料加入原始摘要保留 GraphSAGE 等专有名词

运行：
  cd D:\\Skill Rubri\\agenteval_test01
  python -m pytest tests/test_pipeline_fixes.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ══════════════════════════════════════════════════════════════
# P1 — 数据清洗：领域过滤
# ══════════════════════════════════════════════════════════════
class TestP1DomainFilter:
    """build_index.is_cs_paper() 应正确识别 CS/AI 论文，过滤非目标领域。"""

    def setup_method(self):
        from build_index import is_cs_paper
        self.is_cs_paper = is_cs_paper

    def test_gnn_paper_is_cs(self):
        """GNN 论文应识别为 CS 领域"""
        record = {
            "paper": "GraphSAGE: Inductive Representation Learning on Large Graphs",
            "abstract": "We present GraphSAGE, a general inductive framework for graph neural network.",
            "research_question": "How to learn embeddings for unseen nodes?",
            "methodology": "",
        }
        assert self.is_cs_paper(record) is True

    def test_fluid_dynamics_is_not_cs(self):
        """流体力学论文应被过滤掉"""
        record = {
            "paper": "Turbulent Flow Simulation in Navier-Stokes Equations",
            "abstract": "We study turbulent flow patterns using finite element methods for fluid mechanics.",
            "research_question": "How does viscosity affect turbulence?",
            "methodology": "computational fluid dynamics simulation",
        }
        assert self.is_cs_paper(record) is False

    def test_bert_paper_is_cs(self):
        """BERT 论文应识别为 CS 领域"""
        record = {
            "paper": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, a new language model based on transformer architecture.",
            "research_question": "",
            "methodology": "",
        }
        assert self.is_cs_paper(record) is True

    def test_mathematics_paper_is_not_cs(self):
        """纯数学论文应被过滤掉"""
        record = {
            "paper": "Riemann Hypothesis and Zeta Functions",
            "abstract": "We investigate the distribution of prime numbers using zeta function analysis.",
            "research_question": "Do all non-trivial zeros lie on the critical line?",
            "methodology": "analytic number theory",
        }
        assert self.is_cs_paper(record) is False

    def test_contrastive_learning_is_cs(self):
        """对比学习论文应识别为 CS 领域"""
        record = {
            "paper": "SimCLR: A Simple Framework for Contrastive Learning",
            "abstract": "We propose a contrastive self-supervised learning framework for visual representations.",
            "research_question": "",
            "methodology": "",
        }
        assert self.is_cs_paper(record) is True

    def test_empty_record_is_not_cs(self):
        """空记录应返回 False（无任何 CS 关键词）"""
        record = {"paper": "", "abstract": "", "research_question": "", "methodology": ""}
        assert self.is_cs_paper(record) is False

    def test_cs_keyword_in_methodology_only(self):
        """仅在 methodology 字段有 CS 关键词也应通过"""
        record = {
            "paper": "Efficiency Study",
            "abstract": "",
            "research_question": "",
            "methodology": "We use deep learning to classify results",
        }
        assert self.is_cs_paper(record) is True


# ══════════════════════════════════════════════════════════════
# P2 — Embedding 质量：build_text() 优先用原始摘要
# ══════════════════════════════════════════════════════════════
class TestP2BuildText:
    """build_text() 应优先使用原始摘要，保留专有名词和方法细节。"""

    def setup_method(self):
        from build_index import build_text
        self.build_text = build_text

    def test_uses_abstract_when_available(self):
        """有 abstract 时，输出应包含 abstract 内容"""
        record = {
            "paper": "GraphSAGE Paper",
            "abstract": "We apply GraphSAGE with mean aggregator on OGB-Arxiv benchmark.",
            "research_question": "How to learn node embeddings?",
            "methodology": "graph sampling method",
        }
        text = self.build_text(record)
        assert "GraphSAGE with mean aggregator" in text
        assert "OGB-Arxiv" in text

    def test_abstract_preserves_proper_nouns(self):
        """原始摘要中的专有名词（GraphSAGE、BERT）不应被泛化"""
        record = {
            "paper": "BERT Fine-Tuning",
            "abstract": "We fine-tune BERT-large on SQuAD 2.0 using cross-encoder reranking.",
            "research_question": "uses pre-trained language model",   # GPT 泛化版本
            "methodology": "fine-tuning on reading comprehension",
        }
        text = self.build_text(record)
        assert "BERT-large" in text or "BERT" in text
        assert "SQuAD" in text

    def test_fallback_to_gpt_fields_when_no_abstract(self):
        """无 abstract 时，降级到 GPT 提取字段"""
        record = {
            "paper": "Some Paper",
            "abstract": "",
            "research_question": "How to improve recommendation?",
            "methodology": "collaborative filtering",
            "datasets": "MovieLens",
            "results": "AUC improved by 3%",
            "comment": "solid work",
        }
        text = self.build_text(record)
        assert "collaborative filtering" in text
        assert "MovieLens" in text

    def test_title_always_first(self):
        """标题应始终出现在输出的最前面（最强信号位置）"""
        record = {
            "paper": "Attention Is All You Need",
            "abstract": "We propose the Transformer, a model architecture eschewing recurrence.",
            "research_question": "",
            "methodology": "",
        }
        text = self.build_text(record)
        assert text.startswith("Attention Is All You Need")

    def test_empty_abstract_skips_to_fallback(self):
        """空字符串 abstract 应触发降级逻辑"""
        record = {
            "paper": "Test Paper",
            "abstract": "   ",   # 只有空格，应视为空
            "research_question": "Research Q",
            "methodology": "method",
            "datasets": "",
            "results": "",
            "comment": "",
        }
        text = self.build_text(record)
        # 空格应被 strip，走 fallback 路径
        assert "Research Q" in text or "method" in text

    def test_abstract_much_more_informative_than_gpt(self):
        """原始摘要字数应远多于 GPT 提取字段（信息量更大）"""
        abstract = (
            "We present GraphSAGE, a general inductive framework that leverages node feature information "
            "to efficiently generate node embeddings for previously unseen data. Instead of training "
            "individual embeddings for each node, we learn a function that generates embeddings by "
            "sampling and aggregating features from a node's local neighborhood on the OGB-Arxiv dataset."
        )
        record_with_abstract = {
            "paper": "GraphSAGE",
            "abstract": abstract,
            "research_question": "node embedding",
            "methodology": "graph sampling",
        }
        record_gpt_only = {
            "paper": "GraphSAGE",
            "abstract": "",
            "research_question": "node embedding",
            "methodology": "graph sampling",
        }
        text_abstract = self.build_text(record_with_abstract)
        text_gpt      = self.build_text(record_gpt_only)
        assert len(text_abstract) > len(text_gpt)


# ══════════════════════════════════════════════════════════════
# P3 — Cross-Encoder Rerank
# ══════════════════════════════════════════════════════════════
class TestP3CrossEncoderRerank:
    """rerank_with_cross_encoder() 应正确重排候选列表。"""

    def setup_method(self):
        from hybrid_search import rerank_with_cross_encoder
        self.rerank = rerank_with_cross_encoder

    def test_rerank_returns_same_count(self):
        """精排后候选数量不变"""
        candidates = [
            {"rowid": 1, "text": "GNN node classification graph"},
            {"rowid": 2, "text": "fluid dynamics turbulence simulation"},
            {"rowid": 3, "text": "graph neural network link prediction"},
        ]
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.1, 0.8]   # GNN 相关的高分

        with patch("hybrid_search._get_cross_encoder", return_value=mock_ce):
            results = self.rerank("GNN node classification", candidates)

        assert len(results) == 3

    def test_rerank_changes_order(self):
        """Cross-Encoder 应能改变 Bi-Encoder 的排序"""
        candidates = [
            {"rowid": 1, "text": "fluid dynamics mesh nodes"},   # Bi-Encoder 误判高分
            {"rowid": 2, "text": "GNN node classification"},     # 真正相关
        ]
        mock_ce = MagicMock()
        # Cross-Encoder 正确判断：GNN 相关(0.9) > 流体力学(0.05)
        mock_ce.predict.return_value = [0.05, 0.9]

        with patch("hybrid_search._get_cross_encoder", return_value=mock_ce):
            results = self.rerank("GNN node classification", candidates)

        assert results[0]["rowid"] == 2   # GNN 论文排到第一

    def test_rerank_adds_cross_encoder_score(self):
        """每条结果应包含 cross_encoder_score 字段"""
        candidates = [{"rowid": 1, "text": "deep learning"}]
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.75]

        with patch("hybrid_search._get_cross_encoder", return_value=mock_ce):
            results = self.rerank("neural network", candidates)

        assert "cross_encoder_score" in results[0]
        assert abs(results[0]["cross_encoder_score"] - 0.75) < 1e-5

    def test_rerank_graceful_fallback_when_unavailable(self):
        """CrossEncoder 不可用时，原样返回候选列表（不崩溃）"""
        candidates = [
            {"rowid": 1, "text": "paper A"},
            {"rowid": 2, "text": "paper B"},
        ]
        with patch("hybrid_search._get_cross_encoder", return_value=None):
            results = self.rerank("query", candidates)

        # 降级：返回原始顺序，不报错
        assert len(results) == 2
        assert results[0]["rowid"] == 1

    def test_rerank_empty_candidates(self):
        """空候选列表应直接返回空，不调用模型"""
        mock_ce = MagicMock()
        with patch("hybrid_search._get_cross_encoder", return_value=mock_ce):
            results = self.rerank("GNN", [])
        assert results == []
        mock_ce.predict.assert_not_called()

    def test_rerank_custom_text_field(self):
        """支持自定义文本字段名"""
        candidates = [
            {"rowid": 1, "content": "graph neural network"},
            {"rowid": 2, "content": "biology protein structure"},
        ]
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.8, 0.1]

        with patch("hybrid_search._get_cross_encoder", return_value=mock_ce):
            results = self.rerank("GNN", candidates, text_field="content")

        assert results[0]["rowid"] == 1


# ══════════════════════════════════════════════════════════════
# P4 — 绝对阈值：FAISS 低质量结果兜底
# ══════════════════════════════════════════════════════════════
class TestP4AbsoluteThreshold:
    """search_vector() 和 search_bm25() 应在无相关结果时返回空字典。"""

    def setup_method(self):
        from hybrid_search import search_vector, search_bm25, FAISS_MIN_QUALITY
        self.search_vector    = search_vector
        self.search_bm25      = search_bm25
        self.FAISS_MIN_QUALITY = FAISS_MIN_QUALITY

    def test_faiss_min_quality_threshold_value(self):
        """FAISS_MIN_QUALITY 应在合理范围 [0.05, 0.30]"""
        assert 0.05 <= self.FAISS_MIN_QUALITY <= 0.30, (
            f"FAISS_MIN_QUALITY={self.FAISS_MIN_QUALITY} 超出预期范围，检查是否设置正确"
        )

    def test_vector_search_returns_empty_when_max_below_threshold(self):
        """最高 FAISS 原始分 < FAISS_MIN_QUALITY 时应返回空字典"""
        # 模拟一个索引：所有向量与 query 的余弦 < 0.10（无相关论文）
        mock_index = MagicMock()
        mock_index.ntotal = 3
        # FAISS 原始分全部很低（query 属于语料库没覆盖的领域）
        mock_index.search.return_value = (
            np.array([[0.08, 0.06, 0.04]]),   # 最高仅 0.08，低于阈值
            np.array([[1, 2, 3]])
        )

        import hybrid_search
        original = hybrid_search._faiss_index
        hybrid_search._faiss_index = mock_index
        try:
            result = self.search_vector(np.zeros((1, 1536), dtype="float32"), top_k=3)
        finally:
            hybrid_search._faiss_index = original

        assert result == {}, "最高分低于阈值应返回空字典，而非归一化的假高分"

    def test_vector_search_returns_results_when_above_threshold(self):
        """最高 FAISS 分 ≥ FAISS_MIN_QUALITY 时正常返回归一化结果"""
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.85, 0.60]]),   # 正常相关度
            np.array([[10, 20]])
        )

        import hybrid_search
        original = hybrid_search._faiss_index
        hybrid_search._faiss_index = mock_index
        try:
            result = self.search_vector(np.zeros((1, 1536), dtype="float32"), top_k=2)
        finally:
            hybrid_search._faiss_index = original

        assert len(result) == 2
        assert result[10] == pytest.approx(1.0)   # 最高分归一化为 1.0

    def test_bm25_returns_empty_when_no_hits(self):
        """BM25 完全无命中时应返回空字典"""
        # 构造一批论文，专有名词完全不同
        records = [
            {"rowid": 1, "paper": "fluid dynamics", "abstract": "", "research_question": "",
             "methodology": "navier stokes equations", "datasets": "", "results": "", "comment": ""},
            {"rowid": 2, "paper": "biology genetics", "abstract": "", "research_question": "",
             "methodology": "dna sequencing", "datasets": "", "results": "", "comment": ""},
        ]
        result = self.search_bm25("GraphSAGE node classification GNN", records, top_k=5)
        # 注意：BM25Okapi 对完全没有 term 匹配的 query 会返回全 0 分
        # 期望：max_s <= 0 → 返回 {}
        assert isinstance(result, dict)
        # 若所有分均为 0，max_s=0，应返回 {}
        if all(v == 0 for v in result.values()):
            assert result == {}

    def test_relative_normalization_problem_demonstration(self):
        """演示旧代码问题：即使最高分 0.08，归一化后误报为 1.0"""
        # 旧逻辑（手动复现）：不检查绝对值
        raw_scores = {1: 0.08, 2: 0.06, 3: 0.04}
        max_s = max(raw_scores.values())   # 0.08
        old_normalized = {k: v / max_s for k, v in raw_scores.items()}
        assert old_normalized[1] == pytest.approx(1.0)   # 旧代码误报高分

        # 新逻辑：检查绝对值，0.08 < FAISS_MIN_QUALITY → 返回空
        assert max_s < self.FAISS_MIN_QUALITY, "演示用例的最高分应低于阈值"


# ══════════════════════════════════════════════════════════════
# P5 — BM25 专有名词：build_corpus() 包含原始摘要
# ══════════════════════════════════════════════════════════════
class TestP5BM25ProperNouns:
    """build_corpus() 应包含原始摘要，确保 GraphSAGE 等专有名词可被关键词匹配。"""

    def setup_method(self):
        from hybrid_search import build_corpus, tokenize
        self.build_corpus = build_corpus
        self.tokenize     = tokenize

    def _make_record(self, paper, abstract="", methodology=""):
        return {
            "rowid": 1,
            "paper": paper,
            "abstract": abstract,
            "research_question": "",
            "methodology": methodology,
            "datasets": "",
            "results": "",
            "comment": "",
        }

    def test_proper_noun_in_abstract_appears_in_corpus(self):
        """abstract 中的专有名词应出现在 BM25 语料中"""
        record = self._make_record(
            paper="Graph Learning Paper",
            abstract="We apply GraphSAGE with mean aggregator on the OGB-Arxiv dataset.",
        )
        corpus = self.build_corpus([record])
        assert "GraphSAGE" in corpus[0]

    def test_gpt_rewrites_lose_proper_nouns(self):
        """演示问题：GPT 改写后专有名词丢失"""
        # GPT 改写版：GraphSAGE → "graph sampling method"
        record = self._make_record(
            paper="Graph Learning Paper",
            abstract="",   # 无原始摘要
            methodology="uses graph sampling method on citation dataset",
        )
        corpus = self.build_corpus([record])
        # GPT 改写后，"GraphSAGE" 不在语料中
        assert "GraphSAGE" not in corpus[0]
        assert "graph" in corpus[0].lower()   # 泛化词汇还在

    def test_abstract_fixes_proper_noun_loss(self):
        """有原始摘要时，即使 GPT 字段已泛化，专有名词仍可匹配"""
        record = self._make_record(
            paper="Graph Learning Paper",
            abstract="We apply GraphSAGE with mean aggregator on OGB-Arxiv.",
            methodology="uses graph sampling method",   # GPT 泛化字段
        )
        corpus = self.build_corpus([record])
        assert "GraphSAGE" in corpus[0]
        assert "OGB" in corpus[0] or "ogb" in corpus[0].lower()

    def test_bert_proper_noun_preserved(self):
        """BERT、SQuAD 等专有名词通过 abstract 保留"""
        record = self._make_record(
            paper="BERT Fine-tuning",
            abstract="We fine-tune BERT-large on SQuAD 2.0 and GLUE benchmarks.",
            methodology="fine-tuning pre-trained language model",
        )
        corpus = self.build_corpus([record])
        assert "BERT" in corpus[0]
        assert "SQuAD" in corpus[0]

    def test_tokenize_preserves_model_names(self):
        """tokenize 应保留模型/数据集名称的 token"""
        tokens = self.tokenize("GraphSAGE node2vec OGB-Arxiv BERT-large")
        assert "graphsage" in tokens
        assert "bert" in tokens
        assert "ogb" in tokens or "arxiv" in tokens

    def test_corpus_includes_abstract_field(self):
        """build_corpus 的语料应包含 abstract 字段的内容"""
        abstract_text = "unique_abstract_marker_xyz neural graph convolution"
        record = self._make_record(
            paper="Test Paper",
            abstract=abstract_text,
        )
        corpus = self.build_corpus([record])
        assert "unique_abstract_marker_xyz" in corpus[0]


# ══════════════════════════════════════════════════════════════
# 集成测试：DB abstract 字段迁移
# ══════════════════════════════════════════════════════════════
class TestAbstractColumnMigration:
    """init_db() 应自动为旧数据库添加 abstract 列。"""

    def test_abstract_column_in_migration_list(self):
        """db.init_db() 的 new_columns 列表应包含 abstract 字段"""
        import inspect
        import db
        source = inspect.getsource(db.init_db)
        assert "abstract" in source, (
            "db.init_db() 未包含 abstract 字段的 ALTER TABLE 迁移，请检查 new_columns 列表"
        )

    def test_get_all_papers_returns_abstract_field(self):
        """get_all_papers() 返回的字典应包含 abstract 键（值可为空）"""
        import tempfile, sqlite3, db as db_module

        # 使用临时数据库测试
        original_path = db_module.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_path = f.name

        try:
            db_module.DB_PATH = tmp_path
            db_module.init_db()
            # 插入一条测试记录
            conn = sqlite3.connect(tmp_path)
            conn.execute(
                "INSERT INTO papers (paper, research_question) VALUES (?, ?)",
                ("Test Paper", "Test Q")
            )
            conn.commit()
            conn.close()

            records = db_module.get_all_papers()
            assert len(records) >= 1
            assert "abstract" in records[0], "get_all_papers() 应返回 abstract 字段"
        finally:
            db_module.DB_PATH = original_path
            os.unlink(tmp_path)
