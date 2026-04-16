"""
migrate_to_sqlite.py
---------------------
把 eval_results.json 一次性迁移到 SQLite（papers.db）。
只需运行一次，之后数据由 SQLite 管理。

运行方式：PyCharm 右键 Run
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, upsert_paper, get_all_papers

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EVAL_JSON = os.path.join(DATA_DIR, "eval_results.json")


def run():
    print("=" * 55)
    print("  migrate_to_sqlite.py")
    print("  eval_results.json → papers.db")
    print("=" * 55)

    init_db()
    print("\n  数据库初始化完成")

    with open(EVAL_JSON, encoding="utf-8") as f:
        records = json.load(f)

    print(f"  读取 JSON：{len(records)} 条记录\n")

    for i, r in enumerate(records, 1):
        upsert_paper(r)
        print(f"  [{i}/{len(records)}]  {r.get('paper', '')[:55]}")

    total = len(get_all_papers())
    print(f"\n  完成：papers.db 共 {total} 条记录")
    print(f"  路径：{os.path.join(BASE_DIR, 'data', 'papers.db')}\n")


if __name__ == "__main__":
    run()
