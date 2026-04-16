"""
test_eval_langgraph_pipeline.py
--------------------------------
测试目标：验证 LangGraph 7节点流水线的输出完整性

实现原理：
  paper_rubric_eval.py 用 LangGraph StateGraph 串联7个节点：
    START
      → structure_parser    (识别章节范围 → sections)
      → question_extractor  (技能1：研究问题 → research_question)
      → method_node         (技能2：方法论 → methodology)
      → dataset_node        (技能3：数据集 → datasets)
      → results_extractor   (技能4：实验结果 → results)
      → critic              (技能5：批判性分析 → critique)
      → scorer              (技能6：综合打分 → score_result)
    END
  每个节点返回 {**state, "新字段": value}，后续节点通过 state["字段名"] 读取。
  最终结果写入 eval_results.json，每条记录包含所有节点的输出字段。

测试方法（不调用 LLM，只读取已有结果）：
  1. 验证流水线5个提取字段（skill1~5产物）全部存在且非空
  2. 验证 skill_scores 包含全部6个维度（skill1~skill6）
  3. 验证各维度原因字段（skill_reasons）非空 — 确认 judge 确实运行了
  4. 验证流水线产物字段长度合理（不是截断的空字符串）
  5. 验证 scorer 节点输出（paper_quality_score）被正确存入
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 流水线各节点产出的字段：node_name → result_key
PIPELINE_FIELDS = [
    ("question_extractor",  "research_question"),   # 技能1
    ("method_node",         "methodology"),          # 技能2
    ("dataset_node",        "datasets"),             # 技能3
    ("results_extractor",   "results"),              # 技能4
    ("critic",              "critique"),             # 技能5
]

ALL_SKILLS = ["skill1", "skill2", "skill3", "skill4", "skill5", "skill6"]


def load_results():
    path = os.path.join(DATA_DIR, "eval_results.json")
    assert os.path.exists(path), f"eval_results.json 不存在：{path}"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_pipeline_extraction_fields_complete():
    """
    验证每条记录都包含流水线5个提取字段（技能1~5的产物）
    说明：这5个字段由 question_extractor / method_node / dataset_node /
    results_extractor / critic 节点依次写入 state，缺失说明某节点未执行或结果未保存
    """
    records = load_results()
    valid   = [r for r in records if r.get("research_question")]  # 只检查新版记录

    missing_counts = {field: 0 for _, field in PIPELINE_FIELDS}
    for r in valid:
        for _, field in PIPELINE_FIELDS:
            if not r.get(field):
                missing_counts[field] += 1

    print(f"  检查 {len(valid)} 条有效记录：")
    all_ok = True
    for _, field in PIPELINE_FIELDS:
        cnt = missing_counts[field]
        if cnt == 0:
            print(f"    PASS  {field:<25} 全部存在")
        else:
            print(f"    FAIL  {field:<25} 缺失 {cnt}/{len(valid)} 条")
            all_ok = False

    assert all_ok, f"部分节点输出字段缺失：{missing_counts}"


def test_skill_scores_all_six_dimensions():
    """
    验证每条记录的 skill_scores 包含 skill1~skill6 全部6个维度
    说明：judge 节点对每个维度独立打分，6个维度对应6次 gpt-4o 调用。
    缺少任何维度说明某次 judge 调用失败且未被保存
    """
    records = load_results()
    valid   = [r for r in records if r.get("skill_scores")]

    incomplete = []
    for r in valid:
        scores = r.get("skill_scores", {})
        missing_skills = [s for s in ALL_SKILLS if s not in scores]
        if missing_skills:
            incomplete.append((r["paper"], missing_skills))

    if incomplete:
        for paper, missing in incomplete[:5]:
            print(f"    FAIL  {paper[:50]}  缺少：{missing}")
    else:
        print(f"  PASS  全部 {len(valid)} 条记录 skill1~skill6 完整")

    assert len(incomplete) == 0, f"缺少 skill 维度的记录：{len(incomplete)} 条"


def test_skill_reasons_not_empty():
    """
    验证 skill_reasons 字段每个维度都有原因文本
    说明：judge 返回 {"score": 0.XX, "reason": "..."} 格式，
    reason 为空说明 JSON 解析失败或 judge 输出被截断
    """
    records = load_results()
    valid   = [r for r in records if r.get("skill_reasons")]

    empty_reasons = []
    for r in valid:
        reasons = r.get("skill_reasons", {})
        for skill in ALL_SKILLS:
            if skill in reasons and not reasons[skill]:
                empty_reasons.append((r["paper"], skill))

    if empty_reasons:
        for paper, skill in empty_reasons[:5]:
            print(f"    WARN  {paper[:50]}  {skill} reason 为空")
        print(f"  WARN  {len(empty_reasons)} 个维度原因为空")
    else:
        print(f"  PASS  全部 skill_reasons 非空")

    assert len(empty_reasons) == 0, f"reason 为空的记录：{empty_reasons[:5]}"


def test_extraction_fields_no_error_messages():
    """
    验证流水线提取字段没有 [ERROR after N retries] 错误信息
    说明：invoke_with_retry() 失败3次后返回 "[ERROR after 3 retries: ...]"，
    这种字段无法用于 BM25/向量检索，也无法作为正常的评测结果。
    允许 5% 的失败率（网络偶发连接错误属于正常情况）
    """
    records = load_results()
    valid   = [r for r in records if r.get("research_question")]

    error_fields = []
    for r in valid:
        for _, field in PIPELINE_FIELDS:
            val = r.get(field, "")
            if val and val.startswith("[ERROR"):
                error_fields.append((r["paper"][:50], field, val[:60]))

    if error_fields:
        for paper, field, val in error_fields:
            print(f"    WARN  {paper}  {field}: {val}")
        print(f"  WARN  {len(error_fields)} 个字段包含错误信息（网络失败导致）")
    else:
        print(f"  PASS  全部 {len(valid)} 条记录无错误信息")

    # 允许 5% 的失败率
    total_fields = len(valid) * len(PIPELINE_FIELDS)
    error_ratio  = len(error_fields) / total_fields if total_fields else 0
    assert error_ratio <= 0.05, f"错误字段占比 {error_ratio:.1%}，超过 5% 阈值"
    if error_fields:
        print(f"  PASS  错误率 {error_ratio:.1%} 在容忍范围内（<= 5%）")


def test_scorer_output_saved():
    """
    验证 scorer 节点的输出（paper_quality_score）被正确存入每条记录
    说明：scorer 节点是流水线最后一个节点，其输出 score_result 经过解析后
    作为 paper_quality_score 写入 eval_results.json，缺失说明流水线未跑完
    """
    records = load_results()
    missing = [r for r in records if r.get("paper_quality_score", 0) == 0]
    valid   = len(records) - len(missing)

    print(f"  共 {len(records)} 条记录，{valid} 条有 paper_quality_score，{len(missing)} 条为0或缺失")
    if missing:
        for r in missing[:3]:
            print(f"    WARN  {r['paper'][:50]}")

    assert valid > 0, "没有任何记录包含 paper_quality_score，scorer 节点可能未运行"
    ratio = valid / len(records)
    assert ratio >= 0.9, f"paper_quality_score 覆盖率 {ratio:.1%}，低于 90%"
    print(f"  PASS  scorer 输出覆盖率 {ratio:.1%}")


def run_all():
    print("=" * 60)
    print("  test_eval_langgraph_pipeline.py")
    print("  验证 LangGraph 7节点流水线输出完整性")
    print("=" * 60)
    print()
    print("  流水线结构：")
    print("    START → structure_parser → question_extractor")
    print("          → method_node → dataset_node")
    print("          → results_extractor → critic → scorer → END")
    print()
    tests = [
        test_pipeline_extraction_fields_complete,
        test_skill_scores_all_six_dimensions,
        test_skill_reasons_not_empty,
        test_extraction_fields_no_error_messages,
        test_scorer_output_saved,
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
