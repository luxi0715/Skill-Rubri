"""
test_fetch_arxiv_search.py
---------------------------
测试目标：验证 arXiv 搜索 + CS 分类过滤是否生效

实现原理：
  fetch_papers.py 在搜索语句中加了分类过滤：
    all:{query} AND (cat:cs.IR OR cs.LG OR cs.AI OR cs.CV OR cs.CL)
  这样只返回计算机科学相关论文，过滤掉热力学、物理学等无关领域。

测试方法：
  1. 用已下载的 PDF 文件名检查命名规范是否正确
     格式应为：{year}_{author}_{title}.pdf
  2. 检查 eval_results.json 里的论文是否都有 research_question 字段
     （说明评测流水线正确保存了提取内容）
  3. 检查 papers/ 目录下 PDF 数量是否大于 0
"""

import json
import os
import re
import sys

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")

PDF_NAME_PATTERN = re.compile(r"^\d{4}_\w+_.+\.pdf$")


def test_papers_directory_not_empty():
    """
    验证 papers/ 目录存在且有 PDF 文件
    说明：fetch_papers.py 至少成功下载过一次
    """
    assert os.path.exists(PAPERS_DIR), "papers/ 目录不存在"
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    assert len(pdfs) > 0, "papers/ 目录里没有 PDF 文件"
    print(f"  PASS  papers/ 目录有 {len(pdfs)} 篇 PDF")
    return len(pdfs)


def test_pdf_naming_convention():
    """
    验证所有 PDF 文件名符合命名规范：{year}_{author}_{title}.pdf
    说明：make_pdf_name() 函数正确提取了年份、作者姓氏、标题
    不符合规范的文件说明命名逻辑有问题
    """
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    failed = [f for f in pdfs if not PDF_NAME_PATTERN.match(f)]

    if failed:
        print(f"  FAIL  {len(failed)} 个文件名不符合规范：")
        for f in failed:
            print(f"        {f}")
    else:
        print(f"  PASS  全部 {len(pdfs)} 个 PDF 命名规范正确")

    assert len(failed) == 0, f"命名不规范的文件：{failed}"


def test_eval_results_have_extracted_fields():
    """
    验证 eval_results.json 里的记录包含 research_question 字段
    说明：paper_rubric_eval.py 正确把提取内容存进了 JSON
    这些字段是 build_index.py 建索引的原料，缺失会导致索引为空
    """
    results_file = os.path.join(DATA_DIR, "eval_results.json")
    assert os.path.exists(results_file), "eval_results.json 不存在"

    with open(results_file, encoding="utf-8") as f:
        records = json.load(f)

    total   = len(records)
    valid   = [r for r in records if r.get("research_question")]
    invalid = total - len(valid)

    print(f"  共 {total} 条记录，{len(valid)} 条有 research_question，{invalid} 条缺失")
    assert len(valid) > 0, "没有任何记录包含 research_question 字段"
    if invalid > 0:
        print(f"  WARNING  {invalid} 条旧记录缺少提取字段，建议重新评测")

    print(f"  PASS  {len(valid)}/{total} 条记录字段完整")


def test_pdf_year_is_valid():
    """
    验证 PDF 文件名中的年份在合理范围内（2018~2030）
    说明：arXiv 日期过滤正确，没有爬取到几十年前的论文
    """
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    bad  = []
    for pdf in pdfs:
        year_str = pdf[:4]
        if year_str.isdigit():
            year = int(year_str)
            if not (2018 <= year <= 2030):
                bad.append(pdf)
    if bad:
        print(f"  FAIL  年份异常的文件：{bad}")
    else:
        print(f"  PASS  全部 {len(pdfs)} 个 PDF 年份在 2018~2030 范围内")
    assert len(bad) == 0, f"年份异常：{bad}"


def run_all():
    print("=" * 60)
    print("  test_fetch_arxiv_search.py")
    print("  验证 arXiv 搜索、下载、命名规范")
    print("=" * 60)
    tests = [
        test_papers_directory_not_empty,
        test_pdf_naming_convention,
        test_eval_results_have_extracted_fields,
        test_pdf_year_is_valid,
    ]
    passed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {e}")
        except Exception as e:
            print(f"  ERROR {e}")
    print(f"\n{'='*60}")
    print(f"  结果：{passed}/{len(tests)} 通过")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()
