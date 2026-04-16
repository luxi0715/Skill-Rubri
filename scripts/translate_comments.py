"""
translate_comments.py
----------------------
给 eval_results.json 里每篇论文补翻译：
  comment   → comment_cn
  title（从文件名提取）→ title_cn

运行方式：PyCharm 右键 Run
"""

import json
import os
import sys
import re

sys.path.insert(0, os.path.dirname(__file__))
from paper_rubric_eval import translate_to_chinese

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EVAL_JSON = os.path.join(DATA_DIR, "eval_results.json")


def extract_title(paper: str) -> str:
    """从文件名提取英文标题：2022_Xie_Contrastive_... → Contrastive ..."""
    name  = paper.replace(".pdf", "")
    parts = name.split("_")
    if len(parts) > 2 and re.match(r"^\d{4}$", parts[0]):
        return " ".join(parts[2:])
    return name.replace("_", " ")


def needs_translation(r: dict) -> bool:
    return not r.get("comment_cn") or not r.get("title_cn")


def run():
    print("=" * 55)
    print("  translate_comments.py")
    print("  翻译 comment 和论文标题 → 中文")
    print("=" * 55)

    with open(EVAL_JSON, encoding="utf-8") as f:
        records = json.load(f)

    todo = [r for r in records if needs_translation(r)]
    print(f"\n  共 {len(records)} 篇，需要翻译：{len(todo)} 篇\n")

    for i, r in enumerate(todo, 1):
        paper = r.get("paper", "")
        print(f"  [{i}/{len(todo)}]  {paper[:50]}")
        try:
            to_translate = {
                "comment": r.get("comment", ""),
                "title":   extract_title(paper),
            }
            result = translate_to_chinese(to_translate)
            r["comment_cn"] = result.get("comment", "")
            r["title_cn"]   = result.get("title", "")
            print(f"           标题：{r['title_cn'][:40]}")
        except Exception as e:
            print(f"           失败：{e}")

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n  完成，已保存：{EVAL_JSON}\n")


if __name__ == "__main__":
    run()
