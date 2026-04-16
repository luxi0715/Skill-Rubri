"""
milvus_client.py
-----------------
Milvus 分布式向量数据库客户端。

替代 FAISS IndexIDMap2，适用于 1000万+ 向量场景。

启动 Milvus（需要 Docker）：
  docker compose -f docs/milvus-docker-compose.yml up -d

安装依赖：
  pip install pymilvus

使用方式：
  from milvus_client import MilvusSearch
  ms = MilvusSearch()
  ms.build_index()          # 全量从 SQLite 建索引（一次性）
  results = ms.search(query_vec, top_k=10)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MILVUS_HOST       = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT       = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME   = "papers"
EMBEDDING_DIM     = 1536   # text-embedding-3-small


class MilvusSearch:
    """
    封装 Milvus 集合的创建、插入、搜索操作。
    与 FAISS IndexIDMap2 接口对齐，方便替换。
    """

    def __init__(self):
        self._collection = None

    def _connect(self):
        try:
            from pymilvus import connections
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            return True
        except Exception as e:
            print(f"[Milvus] 连接失败：{e}")
            return False

    def _get_or_create_collection(self):
        from pymilvus import (
            Collection, CollectionSchema, FieldSchema, DataType, utility
        )
        if utility.has_collection(COLLECTION_NAME):
            return Collection(COLLECTION_NAME)

        fields = [
            FieldSchema(name="rowid",     dtype=DataType.INT64,         is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=EMBEDDING_DIM),
        ]
        schema     = CollectionSchema(fields, description="论文向量索引")
        collection = Collection(COLLECTION_NAME, schema)

        # IVF_FLAT 索引：nlist=128（向量数的平方根附近）
        index_params = {
            "metric_type": "IP",          # 内积 = 余弦（向量已归一化）
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128},
        }
        collection.create_index("embedding", index_params)
        print(f"[Milvus] 集合 {COLLECTION_NAME} 创建完成（dim={EMBEDDING_DIM}）")
        return collection

    def build_index(self):
        """
        全量从 SQLite 读取论文，调用 OpenAI 生成 embedding，批量插入 Milvus。
        等价于 build_index.py --mode api，但目标是 Milvus 而非 FAISS。
        """
        if not self._connect():
            return False

        from db import get_all_papers
        from build_index import build_text, build_api

        records = get_all_papers()
        valid   = [r for r in records if r.get("research_question") and r.get("rowid")]
        texts   = [build_text(r) for r in valid]
        rowids  = [r["rowid"] for r in valid]

        print(f"[Milvus] 编码 {len(texts)} 篇论文...")
        embeddings = build_api(texts)   # shape (N, 1536)，已归一化

        collection = self._get_or_create_collection()
        collection.insert([rowids, embeddings.tolist()])
        collection.flush()
        collection.load()

        print(f"[Milvus] 插入完成，共 {collection.num_entities} 条向量")
        return True

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> dict:
        """
        返回 {rowid: 归一化分数}，与 search_vector() 接口一致。
        """
        if not self._connect():
            return {}

        from pymilvus import Collection
        collection = Collection(COLLECTION_NAME)
        collection.load()

        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = collection.search(
            data          = query_vec.tolist(),
            anns_field    = "embedding",
            param         = search_params,
            limit         = top_k,
            output_fields = [],
        )

        raw = {hit.id: hit.score for hit in results[0]}
        max_s = max(raw.values()) if raw else 1.0
        return {rid: v / max_s for rid, v in raw.items()} if max_s > 0 else raw
