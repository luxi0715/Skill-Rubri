"""
translate_excel.py
-------------------
把 eval_results_cn.xlsx 里的英文内容翻译成中文，覆盖保存。

翻译的列：研究问题 / 核心方法 / 数据集 / 实验结果 / 局限性

运行方式：
  PyCharm 右键 Run
"""

import os
import sys
from dotenv import load_dotenv
import openpyxl

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
EXCEL_CN  = os.path.join(DATA_DIR, "eval_results_cn.xlsx")

sys.path.insert(0, os.path.dirname(__file__))
from paper_rubric_eval import translate_to_chinese

# 需要翻译的列名
TRANSLATE_COLS = ["研究问题", "核心方法", "数据集", "实验结果", "局限性","综合评语"]


def is_english(text: str) -> bool:
    """判断内容是否主要是英文（中文字符占比低于10%就认为是英文）"""
    if not text or not text.strip():
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars / len(text) < 0.1


def run():
    print("=" * 55)
    print("  translate_excel.py")
    print("  翻译 eval_results_cn.xlsx 英文内容为中文")
    print("=" * 55)

    assert os.path.exists(EXCEL_CN), f"文件不存在：{EXCEL_CN}"

    wb = openpyxl.load_workbook(EXCEL_CN)
    ws = wb["论文评测"]

    # 读表头，找到需要翻译的列索引
    headers = [cell.value for cell in ws[1]]
    col_indices = {}
    for col_name in TRANSLATE_COLS:
        if col_name in headers:
            col_indices[col_name] = headers.index(col_name) + 1  # openpyxl 列从1开始

    print(f"\n  找到需要翻译的列：{list(col_indices.keys())}")

    rows = list(ws.iter_rows(min_row=2, values_only=False))
    total = len(rows)
    translated = 0

    for i, row in enumerate(rows, 1):
        paper = row[0].value or f"第{i}行"

        # 收集这一行需要翻译的字段
        to_translate = {}
        for col_name, col_idx in col_indices.items():
            val = row[col_idx - 1].value or ""
            if is_english(str(val)):
                to_translate[col_name] = str(val)

        if not to_translate:
            print(f"  [{i}/{total}]  {str(paper)[:45]}  已是中文，跳过")
            continue

        print(f"  [{i}/{total}]  {str(paper)[:45]}  翻译中...")

        # 字段名映射：Excel列名 → json key（translate_to_chinese 用这个key）
        key_map = {
            "研究问题": "research_question",
            "核心方法": "methodology",
            "数据集":   "datasets",
            "实验结果": "results",
            "局限性":   "critique",
            "综合评语": "comment",
        }
        mapped = {key_map[k]: v for k, v in to_translate.items()}

        try:
            cn = translate_to_chinese(mapped)
            reverse_map = {v: k for k, v in key_map.items()}

            for json_key, cn_val in cn.items():
                col_name = reverse_map.get(json_key)
                if col_name and col_name in col_indices:
                    ws.cell(row=i + 1, column=col_indices[col_name]).value = cn_val

            translated += 1
            print(f"           完成")
        except Exception as e:
            print(f"           失败：{e}")

    # 保存
    while True:
        try:
            wb.save(EXCEL_CN)
            break
        except PermissionError:
            input(f"  ⚠ 请先关闭 Excel 文件，关闭后按回车继续...")

    print(f"\n  完成：{translated}/{total} 行已翻译")
    print(f"  已保存：{EXCEL_CN}\n")


if __name__ == "__main__":
    run()
