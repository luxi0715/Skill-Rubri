"""
translate_existing.py
----------------------
对 eval_results.json 里已有的44篇论文补跑翻译。
只翻译，不重新评测，不花 judge（gpt-4o）的钱。

运行方式：
  PyCharm 右键 Run，或：
  conda run -n agenteval_py310 python scripts/translate_existing.py
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_FILE = os.path.join(DATA_DIR, "eval_results.json")

sys.path.insert(0, os.path.dirname(__file__))
from paper_rubric_eval import translate_to_chinese, invoke_with_retry, llm_executor

FIELDS = ["research_question", "methodology", "datasets", "results", "critique"]


def needs_translation(record: dict) -> bool:
    """判断这条记录是否需要翻译（有英文内容但没有中文版本）"""
    has_content = any(record.get(f) for f in FIELDS)
    has_cn      = any(record.get(f + "_cn") for f in FIELDS)
    return has_content and not has_cn


def translate_record(record: dict) -> dict:
    """翻译一条记录，返回加了 _cn 字段的新记录"""
    fields_to_translate = {f: record.get(f, "") for f in FIELDS if record.get(f)}

    # 跳过错误内容
    fields_to_translate = {
        k: v for k, v in fields_to_translate.items()
        if not v.startswith("[ERROR")
    }

    if not fields_to_translate:
        return record

    cn = translate_to_chinese(fields_to_translate)

    updated = dict(record)
    for f in FIELDS:
        updated[f + "_cn"] = cn.get(f, "")
    return updated


def run():
    print("=" * 55)
    print("  translate_existing.py")
    print("  对已有44篇论文补跑中文翻译")
    print("=" * 55)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    todo    = [r for r in records if needs_translation(r)]
    skipped = len(records) - len(todo)

    print(f"\n  共 {len(records)} 条记录")
    print(f"  已有中文：{skipped} 条（跳过）")
    print(f"  需要翻译：{len(todo)} 条\n")

    if not todo:
        print("  全部已翻译，无需处理。")
        return

    updated_map = {r["paper"]: r for r in records}

    for i, record in enumerate(todo, 1):
        paper = record.get("paper", "?")
        print(f"  [{i}/{len(todo)}]  {paper[:55]}")
        try:
            updated = translate_record(record)
            updated_map[paper] = updated
            print(f"         翻译完成")
        except Exception as e:
            print(f"         翻译失败：{e}")

    # 保存回 json（保持原顺序）
    result_list = [updated_map[r["paper"]] for r in records]
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"\n  已保存：{RESULTS_FILE}")
    print(f"  完成：{len(todo)} 篇翻译成功\n")


if __name__ == "__main__":
    run()
