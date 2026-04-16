"""
sync_excel_meta.py
-------------------
把 Excel 里的元数据（领域、年份、会议/期刊、是否开源）
同步到 eval_results.json 和 paper_meta.json。

运行方式：PyCharm 右键 Run
"""

import json
import os
import openpyxl

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EXCEL     = os.path.join(DATA_DIR, "eval_results_cn.xlsx")
EVAL_JSON = os.path.join(DATA_DIR, "eval_results.json")
META_JSON = os.path.join(DATA_DIR, "paper_meta.json")

# Excel列名 → json字段名
COL_MAP = {
    "领域":     "domain",
    "年份":     "year",
    "会议/期刊": "venue",
    "是否开源":  "open_source",
}


def normalize(name: str) -> str:
    """去掉 .pdf 后缀，统一小写，用于模糊匹配"""
    return name.replace(".pdf", "").lower().strip()


def run():
    print("=" * 55)
    print("  sync_excel_meta.py")
    print("  同步 Excel 元数据 → json")
    print("=" * 55)

    # 读 Excel
    wb = openpyxl.load_workbook(EXCEL)
    ws = wb["论文评测"]
    headers = [cell.value for cell in ws[1]]

    col_idx = {}
    for col_name in ["论文名"] + list(COL_MAP.keys()):
        if col_name in headers:
            col_idx[col_name] = headers.index(col_name)

    # 构建 paper → meta 字典
    excel_data = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        paper_name = row[col_idx["论文名"]] or ""
        key = normalize(paper_name)
        if not key:
            continue
        excel_data[key] = {
            json_field: row[col_idx[col_name]]
            for col_name, json_field in COL_MAP.items()
            if col_name in col_idx
        }

    print(f"\n  Excel 读取：{len(excel_data)} 条记录")

    # 更新 eval_results.json
    with open(EVAL_JSON, encoding="utf-8") as f:
        records = json.load(f)

    matched = 0
    for r in records:
        key = normalize(r.get("paper", ""))
        if key in excel_data:
            r.update(excel_data[key])
            matched += 1

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  eval_results.json：{matched}/{len(records)} 条已更新")

    # 更新 paper_meta.json
    with open(META_JSON, encoding="utf-8") as f:
        meta = json.load(f)

    matched_meta = 0
    for m in meta:
        key = normalize(m.get("paper", ""))
        if key in excel_data:
            m.update(excel_data[key])
            matched_meta += 1

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  paper_meta.json：{matched_meta}/{len(meta)} 条已更新")

    print("\n  完成！新增字段：domain / year / venue / open_source\n")


if __name__ == "__main__":
    run()
