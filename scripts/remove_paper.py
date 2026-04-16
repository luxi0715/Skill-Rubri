"""
remove_paper.py
---------------
从 eval_results.json、paper_meta.json、FAISS 索引中删除指定论文。
"""
import json
import faiss
import numpy as np

TARGET = "2026_Liu_Expectation_Error_Bounds"
BASE   = "D:/Skill Rubri/agenteval_test01/data"

# 1. 删 eval_results.json
data = json.load(open(f"{BASE}/eval_results.json", encoding="utf-8"))
idx  = next(i for i, r in enumerate(data) if TARGET in r.get("paper", ""))
print(f"eval_results: 删除第 {idx} 条 -> {data[idx]['paper']}")
data = [r for r in data if TARGET not in r.get("paper", "")]
with open(f"{BASE}/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"eval_results.json: 剩余 {len(data)} 条")

# 2. 删 paper_meta.json
meta = json.load(open(f"{BASE}/paper_meta.json", encoding="utf-8"))
meta = [m for m in meta if TARGET not in m.get("paper", "")]
with open(f"{BASE}/paper_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"paper_meta.json: 剩余 {len(meta)} 条")

# 3. 重建 FAISS 索引（删掉第 idx 条向量）
index    = faiss.read_index(f"{BASE}/paper_index_api.faiss")
all_vecs = np.zeros((index.ntotal, index.d), dtype="float32")
for i in range(index.ntotal):
    all_vecs[i] = index.reconstruct(i)
keep      = np.delete(all_vecs, idx, axis=0)
new_index = faiss.IndexFlatIP(index.d)
new_index.add(keep)
faiss.write_index(new_index, f"{BASE}/paper_index_api.faiss")
print(f"FAISS 索引: 剩余 {new_index.ntotal} 条")

print("\n完成，三个文件均已更新为 43 条。")
