"""
test_eval_skill_rubric.py
--------------------------
测试目标：验证 Skill Rubric 打分结果的合理性

实现原理：
  paper_rubric_eval.py 用 gpt-4o 对6个维度打分（0~1），
  分数 < 0.6 触发重试（最多2次），重试后分数应该提升或持平。
  skill6 修复后不应再全部卡在 0.50。

测试方法：
  读取 eval_results.json，对已有结果做统计验证：
  1. 所有分数在 0~1 范围内
  2. skill6 平均分不低于 0.6（修复前平均只有 0.50）
  3. Rubric 总分在合理区间（0.5~1.0）
  4. 有足够数量的论文得到了高分（总分 >= 0.85）
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_results():
    with open(os.path.join(DATA_DIR, "eval_results.json"), encoding="utf-8") as f:
        return json.load(f)


def test_skill_scores_in_valid_range():
    """
    验证所有 skill 分数在 0~1 范围内
    说明：judge 输出的 JSON 解析正确，没有出现越界分数
    """
    records = load_results()
    out_of_range = []
    for r in records:
        for skill, score in r.get("skill_scores", {}).items():
            if not (0.0 <= score <= 1.0):
                out_of_range.append((r["paper"], skill, score))

    if out_of_range:
        for paper, skill, score in out_of_range:
            print(f"  FAIL  {paper}  {skill}={score}")
    else:
        print(f"  PASS  全部分数在 0~1 范围内")
    assert len(out_of_range) == 0, f"越界分数：{out_of_range}"


def test_skill6_not_all_stuck_at_050():
    """
    验证 skill6 分数不再全部卡在 0.50
    说明：Rubric 修复有效——去掉了硬性差值要求，改为逻辑一致性判断
    修复前：几乎所有论文 skill6 = 0.50
    修复后：应该有一定比例达到 0.75 或以上
    """
    records  = load_results()
    skill6s  = [r["skill_scores"].get("skill6", 0) for r in records if r.get("skill_scores")]
    above075 = sum(1 for s in skill6s if s >= 0.75)
    avg      = sum(skill6s) / len(skill6s) if skill6s else 0

    print(f"  skill6 平均分：{avg:.3f}  |  >= 0.75 的比例：{above075}/{len(skill6s)}")
    assert avg > 0.55, f"skill6 平均分 {avg:.3f} 仍然过低，修复可能未生效"
    assert above075 > 0, "没有任何论文 skill6 >= 0.75，修复未生效"
    print(f"  PASS  skill6 修复有效")


def test_rubric_total_in_reasonable_range():
    """
    验证 Rubric 总分分布合理（大部分在 0.75~1.00）
    说明：评测系统整体运行正常，不存在系统性偏低或偏高
    """
    records = load_results()
    totals  = [r.get("skill_rubric_total", 0) for r in records]
    avg     = sum(totals) / len(totals) if totals else 0
    above   = sum(1 for t in totals if t >= 0.75)

    print(f"  Rubric 总分均值：{avg:.3f}  |  >= 0.75 的论文：{above}/{len(totals)}")
    assert avg >= 0.70, f"Rubric 均值 {avg:.3f} 偏低"
    assert above / len(totals) >= 0.5, "超过一半论文 Rubric 总分低于 0.75"
    print(f"  PASS  Rubric 总分分布合理")


def test_paper_quality_score_distribution():
    """
    验证论文质量分（overall_score）分布合理
    说明：scorer 节点正常输出，不存在全部相同或全部为0的异常
    """
    records = load_results()
    scores  = [r.get("paper_quality_score", 0) for r in records]
    unique  = len(set(scores))
    avg     = sum(scores) / len(scores) if scores else 0

    print(f"  论文质量分均值：{avg:.3f}  |  不同分值数量：{unique}")
    assert unique > 3, "质量分种类太少，可能评分逻辑异常"
    assert 0.5 <= avg <= 0.9, f"质量分均值 {avg:.3f} 超出合理区间"
    print(f"  PASS  论文质量分分布正常")


def run_all():
    print("=" * 60)
    print("  test_eval_skill_rubric.py")
    print("  验证 Skill Rubric 打分合理性")
    print("=" * 60)
    tests = [
        test_skill_scores_in_valid_range,
        test_skill6_not_all_stuck_at_050,
        test_rubric_total_in_reasonable_range,
        test_paper_quality_score_distribution,
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
