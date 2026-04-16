"""
build_index.py
--------------
为 eval_results.json 里的论文构建 FAISS 向量索引。

支持两种 embedding 模式：
  local  — sentence-transformers（all-MiniLM-L6-v2），GPU 加速，完全离线
  api    — OpenAI text-embedding-3-small，联网调用

运行方式：
  python build_index.py --mode local
  python build_index.py --mode api
  python build_index.py --mode both   # 两个都建（默认）
"""

import argparse
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))   # agenteval_test01/
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_FILE = os.path.join(DATA_DIR, "eval_results.json")
META_FILE    = os.path.join(DATA_DIR, "paper_meta.json")
INDEX_LOCAL  = os.path.join(DATA_DIR, "paper_index_local.faiss")
INDEX_API    = os.path.join(DATA_DIR, "paper_index_api.faiss")

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"   # 384 维，英文，~90MB
API_MODEL_NAME   = "text-embedding-3-small"


# ── 领域过滤（Fix P1）────────────────────────────────────
# CS/AI 领域关键词白名单；命中任意一个视为有效论文
_CS_KEYWORDS = {
    "neural", "deep learning", "machine learning", "transformer", "bert", "gpt",
    "graph neural", "gnn", "attention", "embedding", "recommendation", "retrieval",
    "classification", "detection", "segmentation", "generation", "diffusion",
    "reinforcement learning", "contrastive", "self-supervised", "fine-tun",
    "language model", "natural language", "computer vision", "convolutional",
    "recurrent", "lstm", "gradient", "optimization", "federated", "knowledge graph",
    "node classification", "link prediction", "knowledge distillation",
    "object detection", "semantic", "entity", "relation extraction",
}

def is_cs_paper(record: dict) -> bool:
    """判断一条记录是否属于 CS/AI 领域，用于清洗非目标领域数据。"""
    text = " ".join([
        record.get("abstract", ""),
        record.get("paper", ""),
        record.get("research_question", ""),
        record.get("methodology", ""),
    ]).lower()
    return any(kw in text for kw in _CS_KEYWORDS)


# ── 文本拼接（Fix P2/P5）──────────────────────────────────
def build_text(record: dict) -> str:
    """
    把一条评测记录拼成用于 embedding 的文本。

    修复说明：
    - 优先使用原始摘要（abstract）：保留专有名词、方法名、数据集名
    - 无摘要时降级到 GPT 提取字段（兼容存量数据）
    - 标题始终放在最前面，给 Embedding 模型提供最强信号
    """
    title    = record.get("paper", "").strip()
    abstract = record.get("abstract", "").strip()

    if abstract:
        # P2 Fix：用原始摘要，信息量最大，专有名词不丢失
        return f"{title}. {abstract}"

    # 降级：GPT 提取字段（存量数据兜底）
    parts = [
        f"Title: {title}",
        f"Research Question: {record.get('research_question', '')}",
        f"Methodology: {record.get('methodology', '')}",
        f"Datasets: {record.get('datasets', '')}",
        f"Results: {record.get('results', '')}",
        f"Comment: {record.get('comment', '')}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


# ── 加载数据（从 SQLite 读，带 rowid）────────────────────
def load_records() -> list:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from db import get_all_papers
    records = get_all_papers()
    valid   = [r for r in records if r.get("research_question") and r.get("rowid")]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"  ⚠️  跳过 {skipped} 条记录（无提取内容）")

    # P1 Fix：过滤非 CS/AI 领域论文，防止语料库污染
    cs_valid  = [r for r in valid if is_cs_paper(r)]
    filtered  = len(valid) - len(cs_valid)
    if filtered:
        print(f"  [清洗 P1] 过滤领域外论文：{filtered} 篇")

    print(f"  从 SQLite 读取有效记录：{len(cs_valid)} 篇")
    return cs_valid


# ── 本地 Embedding ────────────────────────────────────────
def build_local(texts: list) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  加载本地模型 {LOCAL_MODEL_NAME}（设备：{device}）...")
    model = SentenceTransformer(LOCAL_MODEL_NAME, device=device)

    print(f"  编码 {len(texts)} 篇论文...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # 归一化后内积 = 余弦相似度
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


# ── API Embedding ─────────────────────────────────────────
def build_api(texts: list) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()

    print(f"  调用 OpenAI {API_MODEL_NAME}，共 {len(texts)} 篇...")
    all_vecs = []
    batch_size = 20   # OpenAI 单次最多 2048 个，20篇绰绰有余
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp  = client.embeddings.create(model=API_MODEL_NAME, input=batch)
        vecs  = [item.embedding for item in resp.data]
        all_vecs.extend(vecs)
        print(f"    {min(i + batch_size, len(texts))}/{len(texts)} 完成")

    arr = np.array(all_vecs, dtype="float32")
    # 归一化，与本地模型对齐
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr   = arr / np.clip(norms, 1e-10, None)
    return arr


# ── 建 FAISS 索引（IndexIDMap2 绑定 SQLite rowid）─────────
def build_faiss(embeddings: np.ndarray, rowids: list, index_path: str):
    dim      = embeddings.shape[1]
    flat     = faiss.IndexFlatIP(dim)
    index    = faiss.IndexIDMap2(flat)   # IDMap2 支持 reconstruct
    ids      = np.array(rowids, dtype="int64")
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, index_path)
    print(f"  FAISS 索引已保存：{index_path}（{index.ntotal} 向量，{dim} 维，绑定 rowid）")


# ── 主流程 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "api", "both"], default="api")
    args = parser.parse_args()

    print("=" * 55)
    print("加载评测数据...")
    records = load_records()
    if not records:
        print("没有可用记录，退出。")
        return

    texts  = [build_text(r) for r in records]
    rowids = [r["rowid"] for r in records]

    if args.mode in ("local", "both"):
        print("\n[本地模型 — sentence-transformers]")
        emb_local = build_local(texts)
        build_faiss(emb_local, rowids, INDEX_LOCAL)

    if args.mode in ("api", "both"):
        print("\n[API 模型 — OpenAI]")
        emb_api = build_api(texts)
        build_faiss(emb_api, rowids, INDEX_API)

    print("\n[done] 索引构建完成。")
    print("  API 索引：", INDEX_API)


if __name__ == "__main__":
    main()
