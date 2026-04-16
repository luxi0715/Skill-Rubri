"""
search_papers.py
----------------
语义检索论文库。

使用方式：
  python search_papers.py --query "contrastive learning cold start" --mode local --top 3
  python search_papers.py --query "transformer recommendation" --mode api --top 5
  python search_papers.py --query "diffusion model" --mode both   # 两个索引对比结果
"""

import argparse
import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))   # agenteval_test01/
DATA_DIR    = os.path.join(BASE_DIR, "data")
META_FILE   = os.path.join(DATA_DIR, "paper_meta.json")
INDEX_LOCAL = os.path.join(DATA_DIR, "paper_index_local.faiss")
INDEX_API   = os.path.join(DATA_DIR, "paper_index_api.faiss")

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
API_MODEL_NAME   = "text-embedding-3-small"


# ── 加载元数据 ────────────────────────────────────────────
def load_meta() -> list:
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 编码查询 ──────────────────────────────────────────────
def encode_local(query: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(LOCAL_MODEL_NAME, device=device)
    vec    = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype("float32")


def encode_api(query: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    resp   = client.embeddings.create(model=API_MODEL_NAME, input=[query])
    vec    = np.array([resp.data[0].embedding], dtype="float32")
    norm   = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.clip(norm, 1e-10, None)


# ── 检索 ──────────────────────────────────────────────────
def search(index_path: str, query_vec: np.ndarray, meta: list, top_k: int):
    index    = faiss.read_index(index_path)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append({
            "rank":                rank,
            "paper":               m["paper"],
            "similarity":          round(float(score), 4),
            "paper_quality_score": m["paper_quality_score"],
            "skill_rubric_total":  m["skill_rubric_total"],
            "comment":             m["comment"][:120] + "..." if len(m["comment"]) > 120 else m["comment"],
        })
    return results


def print_results(results: list, label: str):
    print(f"\n{'─'*55}")
    print(f"  [{label}] 检索结果")
    print(f"{'─'*55}")
    for r in results:
        print(f"  #{r['rank']}  {r['paper']}")
        print(f"      相似度：{r['similarity']:.4f}  |  质量分：{r['paper_quality_score']}  |  Rubric：{r['skill_rubric_total']:.2f}")
        print(f"      {r['comment']}")
        print()


# ── 主流程 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="检索词（英文或中文均可）")
    parser.add_argument("--mode",  choices=["local", "api", "both"], default="local")
    parser.add_argument("--top",   type=int, default=3, help="返回 top-k 结果")
    args = parser.parse_args()

    meta = load_meta()
    print(f"\n查询：{args.query}")
    print(f"模式：{args.mode}  |  top-{args.top}")

    if args.mode in ("local", "both"):
        if not os.path.exists(INDEX_LOCAL):
            print("本地索引不存在，请先运行 python build_index.py --mode local")
        else:
            vec     = encode_local(args.query)
            results = search(INDEX_LOCAL, vec, meta, args.top)
            print_results(results, "本地模型 sentence-transformers")

    if args.mode in ("api", "both"):
        if not os.path.exists(INDEX_API):
            print("API 索引不存在，请先运行 python build_index.py --mode api")
        else:
            vec     = encode_api(args.query)
            results = search(INDEX_API, vec, meta, args.top)
            print_results(results, "OpenAI text-embedding-3-small")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # 右键 Run 时执行测试查询
        meta = load_meta()
        test_queries = [
            "contrastive learning recommendation",
            "reinforcement learning reward",
            "neural network image detection",
        ]
        for query in test_queries:
            print(f"\n{'='*55}")
            print(f"查询：{query}")
            if os.path.exists(INDEX_API):
                vec     = encode_api(query)
                results = search(INDEX_API, vec, meta, top_k=3)
                print_results(results, "OpenAI API")
            else:
                print("API 索引不存在，请先运行 build_index.py")
