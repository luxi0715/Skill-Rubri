"""
test_search_hybrid_fusion.py
-----------------------------
测试目标：验证 BM25 + 向量混合检索的融合效果

实现原理：
  hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
  向量检索擅长语义模糊匹配，BM25 擅长专有名词精确匹配。
  两者融合后，在语义查询和关键词查询场景均应优于单一模式。

测试方法：
  对每个查询设置"期望命中论文"，
  检查三种模式（BM25 / Vector / Hybrid）是否都能在 top-3 命中。
  统计并对比三种模式的命中率。
"""

import json
import os
import re
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── 复用 hybrid_search 的核心函数 ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from hybrid_search import (
    load_meta, load_eval_results, build_corpus,
    tokenize, search_bm25, encode_api, search_vector,
    hybrid_scores, top_k_results
)

# ── 测试用例 ──────────────────────────────────────────────
# 格式：(查询词, 期望出现在结果里的论文名关键词, 测试说明)
TEST_CASES = [
    (
        "contrastive learning sequential recommendation",
        "Xie_Contrastive_Learning_for_Sequential",
        "精确领域查询 → 应命中 CL4SRec 论文"
    ),
    (
        "cold start recommendation sparse user",
        "Hu_Contrastive_Learning_for_Cold_Start",
        "冷启动场景查询 → 应命中 Cold Start 论文"
    ),
    (
        "NeuralUCB LLM routing cost reward",
        "Tsai_Reward-Based_Online_LLM_Routing",
        "专有名词查询 → BM25 应精准命中 NeuralUCB 论文"
    ),
    (
        "graph augmentation recommendation simplified",
        "Yu_Are_Graph_Augmentations_Necessary",
        "图增强方法查询 → 应命中图增强论文"
    ),
    (
        "reinforcement learning skill bank agent planning",
        "Tu_Dynamic_Dual-Granularity_Skill_Bank",
        "技能库强化学习查询 → 应命中 D2Skill 论文"
    ),
]

TOP_K  = 3
ALPHA  = 0.7


def hit(results: list, keyword: str) -> bool:
    """检查 keyword 是否出现在任意结果的论文名里"""
    return any(keyword.lower() in r["paper"].lower() for r in results)


def run_tests():
    print("=" * 65)
    print("  混合检索测试")
    print("=" * 65)

    meta    = load_meta()
    records = load_eval_results()
    valid   = [r for r in records if r.get("research_question")]
    corpus  = build_corpus(valid)

    total   = len(TEST_CASES)
    passed  = {"bm25": 0, "vector": 0, "hybrid": 0}

    for i, (query, expect_kw, desc) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{total}] {desc}")
        print(f"  查询：{query}")
        print(f"  期望包含：{expect_kw}")

        bm25_s = search_bm25(query, corpus, TOP_K)
        vec_s  = search_vector(encode_api(query), TOP_K)
        hy_s   = hybrid_scores(bm25_s, vec_s, ALPHA)

        for mode, scores, label in [
            ("bm25",   bm25_s, "BM25   "),
            ("vector", vec_s,  "Vector "),
            ("hybrid", hy_s,   "Hybrid "),
        ]:
            results = top_k_results(scores, meta, TOP_K, label)
            ok      = hit(results, expect_kw)
            status  = "PASS" if ok else "FAIL"
            if ok:
                passed[mode] += 1
            top1 = results[0]["paper"] if results else "N/A"
            print(f"  {label}  [{status}]  #1 -> {top1[:55]}")

    print("\n" + "=" * 65)
    print("  测试结果汇总")
    print("=" * 65)
    for mode, label in [("bm25","BM25  "),("vector","Vector"),("hybrid","Hybrid")]:
        rate = passed[mode] / total * 100
        bar  = "#" * passed[mode] + "-" * (total - passed[mode])
        print(f"  {label}  [{bar}]  {passed[mode]}/{total}  ({rate:.0f}%)")
    print()

    best = max(passed, key=passed.get)
    print(f"  本次最优模式：{best.upper()}（{passed[best]}/{total} 命中）")
    print()


if __name__ == "__main__":
    run_tests()
