"""
patch_chinese_fields.py
------------------------
批量补全 title_cn 和 comment_cn 字段。

对 eval_results.json 中缺少这两个中文字段的论文，
用 GPT-4o-mini 翻译并写回 JSON + SQLite。

运行：
    python scripts/patch_chinese_fields.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
EVAL_FILE  = os.path.join(BASE_DIR, "data", "eval_results.json")
sys.path.insert(0, os.path.dirname(__file__))
from db import upsert_paper

client = OpenAI()


def filename_to_title(paper: str) -> str:
    """从文件名提取英文标题（去掉年份和作者）"""
    name  = paper.replace(".pdf", "")
    parts = name.split("_")
    if len(parts) > 2 and parts[0].isdigit():
        return " ".join(parts[2:])
    return name.replace("_", " ")


def translate_title(en_title: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "Translate this academic paper title to concise Chinese (keep proper nouns in English if needed). "
                "Output ONLY the translated title, nothing else.\n\n" + en_title
            )
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def translate_comment(en_comment: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "Translate the following academic paper review/comment to Chinese. "
                "Output ONLY the translated text, nothing else.\n\n" + en_comment
            )
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def main():
    with open(EVAL_FILE, encoding="utf-8") as f:
        data = json.load(f)

    need_patch = [r for r in data if not r.get("title_cn") or not r.get("comment_cn")]
    print(f"共 {len(data)} 篇，需补全中文字段：{len(need_patch)} 篇")

    changed = 0
    for i, record in enumerate(data):
        paper = record.get("paper", "")
        need_title   = not record.get("title_cn")
        need_comment = not record.get("comment_cn")

        if not need_title and not need_comment:
            continue

        print(f"[{i+1}/{len(data)}] {paper}")

        try:
            if need_title:
                en_title = filename_to_title(paper)
                record["title_cn"] = translate_title(en_title)
                print(f"  title_cn: {record['title_cn']}")
                time.sleep(0.3)

            if need_comment and record.get("comment"):
                record["comment_cn"] = translate_comment(record["comment"])
                print(f"  comment_cn: {record['comment_cn'][:60]}…")
                time.sleep(0.3)

            # 写回 SQLite
            upsert_paper(record)
            changed += 1

        except Exception as e:
            print(f"  ⚠ 失败：{e}")
            time.sleep(2)

    # 写回 JSON
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 补全完成，共更新 {changed} 篇，已写回 JSON + SQLite")


if __name__ == "__main__":
    main()
