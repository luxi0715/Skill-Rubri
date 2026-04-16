"""
test_search_vector_api.py
--------------------------
测试目标：验证向量语义检索的召回率

实现原理：
  search_papers.py 把查询词转成 embedding 向量，
  用 FAISS IndexFlatIP 做余弦相似度搜索。
  语义检索的优势：即使用词不同，语义相近的论文也能被召回。
  例如："sparse user data" 能匹配到描述 cold start 的论文。

测试方法：
  对每个查询设置期望命中的论文（有明确答案），
  验证 top-3 结果中包含期望论文，并打印相似度分数。
"""

import json
import os
import sys
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
META_FILE = os.path.join(DATA_DIR, "paper_meta.json")
INDEX_API = os.path.join(DATA_DIR, "paper_index_api.faiss")
API_MODEL = "text-embedding-3-small"

# 测试用例：(查询词, 期望命中的论文名关键词, 场景说明)
VECTOR_TEST_CASES = [
    (
        "sequential recommendation contrastive self-supervised",
        "Xie_Contrastive_Learning_for_Sequential",
        "语义近义词查询：self-supervised 能匹配到 contrastive learning 论文"
    ),
    (
        "sparse interaction cold start new user",
        "Hu_Contrastive_Learning_for_Cold_Start",
        "语义扩展：用冷启动场景描述查询相关论文"
    ),
    (
        "online learning bandit algorithm exploration exploitation",
        "Tsai_Reward-Based_Online_LLM_Routing",
        "语义泛化：bandit 算法描述能匹配 NeuralUCB 论文"
    ),
    (
        "graph neural network data augmentation",
        "Yu_Are_Graph_Augmentations_Necessary",
        "图神经网络增强方法查询"
    ),
]


def encode_query(query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp   = client.embeddings.create(model=API_MODEL, input=[query])
    vec    = np.array([resp.data[0].embedding], dtype="float32")
    norm   = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.clip(norm, 1e-10, None)


def search_top_k(query_vec: np.ndarray, k: int = 3):
    index = faiss.read_index(INDEX_API)
    scores, indices = index.search(query_vec, k)
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(meta):
            results.append({"paper": meta[idx]["paper"], "similarity": float(score)})
    return results


def test_semantic_retrieval_recall():
    """
    验证语义检索对4个不同场景的召回率
    说明：相似度 > 0.4 表示语义相关，
    命中表示期望论文出现在 top-3 结果中
    """
    passed = 0
    for query, expect_kw, desc in VECTOR_TEST_CASES:
        print(f"\n  场景：{desc}")
        print(f"  查询：{query}")
        vec     = encode_query(query)
        results = search_top_k(vec, k=3)

        hit = any(expect_kw.lower() in r["paper"].lower() for r in results)
        top1_sim = results[0]["similarity"] if results else 0

        for r in results:
            marker = "<-- 命中" if expect_kw.lower() in r["paper"].lower() else ""
            print(f"    [{r['similarity']:.4f}] {r['paper'][:55]} {marker}")

        if hit:
            passed += 1
            print(f"  PASS")
        else:
            print(f"  FAIL  未找到：{expect_kw}")

    total = len(VECTOR_TEST_CASES)
    assert passed == total, f"召回率 {passed}/{total}，部分查询未命中"
    print(f"\n  总体 PASS  {passed}/{total} 场景全部命中")


def test_similarity_scores_are_positive():
    """
    验证相似度分数为正数（归一化向量的内积应 > 0）
    说明：如果分数为负，说明向量方向相反，归一化可能有问题
    """
    vec     = encode_query("contrastive learning recommendation")
    results = search_top_k(vec, k=5)
    negatives = [r for r in results if r["similarity"] < 0]

    print(f"  top-5 相似度：{[round(r['similarity'],4) for r in results]}")
    assert len(negatives) == 0, f"出现负相似度：{negatives}"
    print(f"  PASS  所有相似度为正数")


def run_all():
    print("=" * 60)
    print("  test_search_vector_api.py")
    print("  验证向量语义检索召回率")
    print("=" * 60)
    tests = [
        test_semantic_retrieval_recall,
        test_similarity_scores_are_positive,
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
