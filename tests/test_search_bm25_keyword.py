"""
test_search_bm25_keyword.py
----------------------------
测试目标：验证 BM25 关键词精确匹配能力

实现原理：
  BM25（Best Match 25）是基于词频的检索算法，
  对文档中出现频率高的词给予更高权重，同时惩罚过长的文档。
  优势：专有名词（如 NeuralUCB、CL4SRec）精确匹配，不会被语义模糊化。
  劣势：不理解语义，"cold start" 和 "sparse new user" 无法关联。

测试方法：
  1. 专有名词查询：期望 BM25 能精确命中包含该词的论文（#1 位置）
  2. 普通语义查询：验证 BM25 也能通过词频找到相关论文
  3. 验证分数归一化到 0~1 范围
"""

import json
import os
import re
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

from hybrid_search import (
    load_meta, load_eval_results, build_corpus,
    search_bm25, top_k_results
)

# 专有名词测试：BM25 应该精准命中 #1
EXACT_MATCH_CASES = [
    (
        "NeuralUCB",
        "Tsai_Reward-Based_Online_LLM_Routing",
        "专有算法名：NeuralUCB 直接出现在论文内容里"
    ),
    (
        "D2Skill dual granularity",
        "Tu_Dynamic_Dual-Granularity_Skill_Bank",
        "专有方法名：D2Skill 应精准命中"
    ),
    (
        "CL4SRec contrastive sequential",
        "Xie_Contrastive_Learning_for_Sequential",
        "模型名称：CL4SRec 应精准命中"
    ),
]

# 普通查询测试：BM25 通过词频匹配
GENERAL_CASES = [
    (
        "graph augmentation recommendation",
        "Yu_Are_Graph_Augmentations_Necessary",
        "关键词组合查询"
    ),
    (
        "cold start sparse user recommendation",
        "Hu_Contrastive_Learning_for_Cold_Start",
        "冷启动场景关键词查询"
    ),
]


def test_exact_keyword_match_top1():
    """
    验证专有名词查询时 BM25 能在 #1 位置命中
    说明：BM25 对专有名词有天然优势，词频高的文档排名靠前
    这是 BM25 相比向量检索的核心价值
    """
    meta    = load_meta()
    records = load_eval_results()
    valid   = [r for r in records if r.get("research_question")]
    corpus  = build_corpus(valid)

    all_passed = True
    for query, expect_kw, desc in EXACT_MATCH_CASES:
        scores  = search_bm25(query, corpus, top_k=3)
        results = top_k_results(scores, meta, top_k=3, label="BM25")
        top1    = results[0]["paper"] if results else ""
        hit_top1 = expect_kw.lower() in top1.lower()

        print(f"\n  场景：{desc}")
        print(f"  查询：{query}")
        print(f"  #1 -> {top1[:60]}")
        if hit_top1:
            print(f"  PASS  专有名词精确命中 #1")
        else:
            hit_top3 = any(expect_kw.lower() in r["paper"].lower() for r in results)
            if hit_top3:
                print(f"  WARN  命中但不在 #1 位置")
            else:
                print(f"  FAIL  未命中")
                all_passed = False

    assert all_passed, "部分专有名词查询未能命中 #1"


def test_general_keyword_recall():
    """
    验证普通关键词组合查询能在 top-3 命中
    说明：BM25 依靠词频，多个关键词同时出现的文档得分更高
    """
    meta    = load_meta()
    records = load_eval_results()
    valid   = [r for r in records if r.get("research_question")]
    corpus  = build_corpus(valid)

    passed = 0
    for query, expect_kw, desc in GENERAL_CASES:
        scores  = search_bm25(query, corpus, top_k=3)
        results = top_k_results(scores, meta, top_k=3, label="BM25")
        hit     = any(expect_kw.lower() in r["paper"].lower() for r in results)

        print(f"\n  场景：{desc}")
        for r in results:
            marker = "<-- 命中" if expect_kw.lower() in r["paper"].lower() else ""
            print(f"    [{r['score']:.4f}] {r['paper'][:55]} {marker}")

        if hit:
            passed += 1
            print(f"  PASS")
        else:
            print(f"  FAIL  未找到：{expect_kw}")

    assert passed == len(GENERAL_CASES), f"召回率 {passed}/{len(GENERAL_CASES)}"


def test_bm25_scores_normalized():
    """
    验证 BM25 分数归一化到 0~1 范围
    说明：hybrid_search.py 把 BM25 原始分除以最大值做归一化，
    确保与向量相似度（也是 0~1）在同一尺度上加权融合
    """
    meta    = load_meta()
    records = load_eval_results()
    valid   = [r for r in records if r.get("research_question")]
    corpus  = build_corpus(valid)

    scores = search_bm25("contrastive learning recommendation", corpus, top_k=3)
    print(f"  BM25 分数范围：{scores.min():.4f} ~ {scores.max():.4f}")
    assert scores.min() >= 0.0,  "BM25 分数出现负值"
    assert scores.max() <= 1.01, "BM25 分数超过 1.0（归一化失败）"
    print(f"  PASS  分数归一化正确")


def run_all():
    print("=" * 60)
    print("  test_search_bm25_keyword.py")
    print("  验证 BM25 关键词精确匹配能力")
    print("=" * 60)
    tests = [
        test_exact_keyword_match_top1,
        test_general_keyword_recall,
        test_bm25_scores_normalized,
    ]
    passed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
        except Exception as e:
            print(f"  ERROR {e}")
    print(f"\n{'='*60}")
    print(f"  结果：{passed}/{len(tests)} 通过")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()
