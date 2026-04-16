"""
test_faiss_inmemory.py
----------------------
验证 Step 1：FAISS 索引常驻内存改造

测试点：
  1. 模块导入后 _faiss_index 不为 None
  2. _faiss_index 是同一个对象（不会每次重新加载）
  3. search_vector 返回格式正确（dict，key=rowid int，value=float 0~1）
  4. search_vector 两次调用返回相同结果（内存稳定性）
  5. 索引向量数量 > 0

运行：
  cd D:\Skill Rubri\agenteval_test01
  python -m pytest tests/test_faiss_inmemory.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import hybrid_search as hs


def test_index_loaded():
    """索引对象不为 None"""
    assert hs._faiss_index is not None, "FAISS 索引未加载，检查 paper_index_api.faiss 是否存在"


def test_index_is_singleton():
    """两次访问是同一个对象（内存常驻，不重复读磁盘）"""
    idx1 = hs._faiss_index
    idx2 = hs._faiss_index
    assert idx1 is idx2


def test_index_has_vectors():
    """索引包含向量"""
    assert hs._faiss_index.ntotal > 0, "索引为空"


def test_search_vector_returns_dict():
    """search_vector 返回 dict"""
    dim = hs._faiss_index.d
    query_vec = np.random.rand(1, dim).astype("float32")
    # 归一化
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    result = hs.search_vector(query_vec, top_k=5)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_search_vector_scores_in_range():
    """所有分数在 [0, 1] 范围内"""
    dim = hs._faiss_index.d
    query_vec = np.random.rand(1, dim).astype("float32")
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    result = hs.search_vector(query_vec, top_k=10)
    for rid, score in result.items():
        assert isinstance(rid, int), f"rowid 应为 int，得到 {type(rid)}"
        assert 0.0 <= score <= 1.0 + 1e-6, f"score 超出范围：{score}"


def test_search_vector_deterministic():
    """同一查询向量两次调用结果一致"""
    dim = hs._faiss_index.d
    query_vec = np.ones((1, dim), dtype="float32")
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    r1 = hs.search_vector(query_vec, top_k=5)
    r2 = hs.search_vector(query_vec, top_k=5)
    assert r1 == r2, "同一查询两次结果不一致"


def test_max_score_is_one():
    """归一化后最高分应为 1.0"""
    dim = hs._faiss_index.d
    query_vec = np.random.rand(1, dim).astype("float32")
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    result = hs.search_vector(query_vec, top_k=hs._faiss_index.ntotal)
    if result:
        assert abs(max(result.values()) - 1.0) < 1e-5, "最高分不为 1.0，归一化异常"
