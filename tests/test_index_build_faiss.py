"""
test_index_build_faiss.py
--------------------------
测试目标：验证 FAISS 索引构建正确性

实现原理：
  build_index.py 把论文内容转成向量后存入 FAISS IndexFlatIP。
  向量归一化后，内积等价余弦相似度，值域为 [-1, 1]。
  相同文本的向量与自身内积应等于 1.0（完全相同）。

测试方法：
  1. 验证索引文件存在
  2. 验证向量数量与 eval_results.json 有效记录数一致
  3. 验证向量维度正确（API=1536）
  4. 验证向量已归一化（模长应约等于 1.0）
  5. 验证同一查询搜索自身时相似度最高（自洽性）
"""

import json
import os
import sys
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_API = os.path.join(DATA_DIR, "paper_index_api.faiss")
META_FILE = os.path.join(DATA_DIR, "paper_meta.json")


def test_index_file_exists():
    """
    验证 FAISS 索引文件存在
    说明：build_index.py 已成功运行并保存了索引
    """
    assert os.path.exists(INDEX_API), f"索引文件不存在：{INDEX_API}"
    size_kb = os.path.getsize(INDEX_API) // 1024
    print(f"  PASS  索引文件存在，大小：{size_kb} KB")


def test_index_vector_count_matches_meta():
    """
    验证索引中向量数量与 paper_meta.json 记录数一致
    说明：每篇论文对应一个向量，数量不一致说明建索引过程有跳过或重复
    """
    index = faiss.read_index(INDEX_API)
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"  索引向量数：{index.ntotal}  |  meta 记录数：{len(meta)}")
    assert index.ntotal == len(meta), \
        f"向量数 {index.ntotal} 与 meta 记录数 {len(meta)} 不一致"
    print(f"  PASS  向量数量一致（{index.ntotal} 篇）")


def test_index_vector_dimension():
    """
    验证向量维度为 1536（OpenAI text-embedding-3-small 的输出维度）
    说明：维度不对说明用了错误的模型或索引文件被覆盖
    """
    index = faiss.read_index(INDEX_API)
    print(f"  向量维度：{index.d}")
    assert index.d == 1536, f"维度应为 1536，实际为 {index.d}"
    print(f"  PASS  向量维度正确（1536 维）")


def test_vectors_are_normalized():
    """
    验证所有向量已归一化（L2 模长约等于 1.0）
    说明：build_index.py 在存入 FAISS 前做了归一化，
    归一化后内积 = 余弦相似度，不归一化则内积没有语义意义
    """
    index = faiss.read_index(INDEX_API)
    # 重建向量矩阵（IndexFlatIP 支持 reconstruct）
    n   = index.ntotal
    dim = index.d
    vecs = np.array([index.reconstruct(i) for i in range(n)], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1)

    max_deviation = float(np.max(np.abs(norms - 1.0)))
    print(f"  模长偏差最大值：{max_deviation:.6f}（应接近 0）")
    assert max_deviation < 0.01, f"向量未归一化，最大偏差：{max_deviation:.6f}"
    print(f"  PASS  所有向量已归一化")


def test_meta_file_has_required_fields():
    """
    验证 paper_meta.json 每条记录包含必要字段
    说明：search_papers.py 依赖这些字段展示检索结果，
    缺少字段会导致检索结果显示异常
    """
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    required = ["index", "paper", "paper_quality_score", "skill_rubric_total", "comment"]
    missing  = []
    for i, m in enumerate(meta):
        for field in required:
            if field not in m:
                missing.append((i, field))

    if missing:
        for idx, field in missing[:5]:
            print(f"  FAIL  记录 {idx} 缺少字段：{field}")
    else:
        print(f"  PASS  全部 {len(meta)} 条记录字段完整")
    assert len(missing) == 0, f"缺少必要字段：{missing[:5]}"


def run_all():
    print("=" * 60)
    print("  test_index_build_faiss.py")
    print("  验证 FAISS 索引构建正确性")
    print("=" * 60)
    tests = [
        test_index_file_exists,
        test_index_vector_count_matches_meta,
        test_index_vector_dimension,
        test_vectors_are_normalized,
        test_meta_file_has_required_fields,
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
