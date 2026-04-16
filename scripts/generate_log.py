import datetime
import os

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "项目日志.txt")

lines = []

def h1(title):
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)

def h2(title):
    lines.append("")
    lines.append(f"── {title} ──")

def para(text):
    for line in text.strip().split("\n"):
        lines.append(line)

def bullet(items):
    for item in items:
        lines.append(f"  • {item}")

# ── 标题 ──
lines.append("AI 论文评测系统 · 项目日志")
lines.append(f"最后更新：{now}")
lines.append("项目路径：D:\\Skill Rubri\\agenteval_test01\\")
lines.append("-" * 60)

# ══ 一、项目现状 ══
h1("一、项目现状（已完成）")
bullet([
    "LangGraph 7节点评测流水线（structure_parser → question_extractor → method_node → dataset_node → results_extractor → critic → scorer）",
    "Skill Rubric 逐项评分（skill1~6，gpt-4o 作为 judge，gpt-4o-mini 执行）",
    "低分重试闭环：Skill < 0.6 → 生成反馈 → 重做节点 → 重新评分（最多2次）",
    "综合评语自动生成（3~5句，写入 Excel）",
    "双语 Excel 导出（eval_results_en.xlsx 两张表 / eval_results_cn.xlsx 一张表）",
    "批量评测 evaluate_folder()，自动跳过已评测论文",
    "论文质量分排序（Skill6 overall_score），不是 Rubric 分",
    "LangGraph state 修复（每个节点返回 {**state, 'new_key': value}）",
    "arXiv API 爬取脚本（fetch_papers.py），热门+经典双渠道，指数退避限速处理",
    "topics_config.json 配置驱动，20个预设领域，enabled true/false 控制",
    "PDF 命名规范：{year}_{first_author}_{title_short}.pdf",
    "arXiv 搜索加分类过滤（cs.IR / cs.LG / cs.AI / cs.CV / cs.CL），修复无关论文问题",
    "skill6 Rubric 修复：去掉硬性差值要求，改为'评分有逻辑支撑、与分析内容一致'",
])

# ══ 二、首次完整运行结果 ══
h1("二、首次完整运行结果（2026-04-01）")
para("""\
领域：contrastive learning recommendation
下载论文：19篇新PDF（1篇因网络中断失败）
评测论文：22篇（含历史3篇）
运行状态：全部完成，无崩溃""")

h2("论文质量分分布（overall_score）")
scores = [
    ("2022_Xie_Contrastive_Learning_for_Sequential_Reco", "0.72"),
    ("2025_Hu_Contrastive_Learning_for_Cold_Start_Reco",  "0.65"),
    ("2025_Qu_Intent-aware_Diffusion_with_Contrastive_",  "0.75"),
    ("2026_Savani_Stepwise_Credit_Assignment_for_GRPO",   "0.75"),
    ("2026_Jin_SonoWorld_From_One_Image_to_3D_Audio",     "0.72"),
    ("2026_Feng_DreamLite_Lightweight_On-Device",         "0.68"),
    ("2026_Schroeder_SOLE-R1_Video-Language_Reasoning",   "0.75"),
    ("2026_Tu_Dynamic_Dual-Granularity_Skill_Bank",       "0.75"),
    ("其他14篇", "0.58 ~ 0.72"),
]
lines.append(f"  {'论文':<52} {'质量分'}")
lines.append("  " + "-" * 62)
for name, score in scores:
    lines.append(f"  {name:<52} {score}")

h2("Skill Rubric 总分分布（Agent行为质量）")
para("""\
平均 Rubric 总分：约 0.85 / 1.00
最高：0.92（DreamLite、SonoWorld）
最低：0.75（Contrastive Learning for Sequential Rec）
重试有效案例：skill4 从 0.25 重试后达到 1.00；skill5 网络中断0.00重试后恢复 0.75""")

# ══ 三、发现的系统性问题及修复 ══
h1("三、系统性问题及修复")

h2("问题1：arXiv 搜索结果不相关（已修复）")
para("""\
现象：搜索 'contrastive learning recommendation' 匹配到大量无关论文
  - 热力学论文（Thermomechanics Learning）
  - 物理学论文（Microscopic Mechanism）
  - 心脏建模论文（four-chamber shape model）
原因：arXiv all: 字段匹配过于宽泛，只要正文出现关键词就命中
修复：在搜索语句加 cs 分类过滤
  (cat:cs.IR OR cat:cs.LG OR cat:cs.AI OR cat:cs.CV OR cat:cs.CL)
状态：已修复（fetch_papers.py）""")

h2("问题2：skill6 综合打分永远卡在 0.50（已修复）")
para("""\
现象：几乎所有论文 skill6 固定为 0.50，重试2次也无法改变
原因：Rubric 要求维度分差 > 0.15，学术论文天然集中在 0.65~0.85，差值只有 0.10~0.15
修复：去掉硬性差值要求，改为"评分是否有逻辑支撑、与分析内容一致"
状态：已修复（paper_rubric_eval.py SKILL_RUBRICS["skill6"]）""")

# ══ 四、面试题回答 ══
h1("四、面试题回答（以项目为证据）")

qa_list = [
    (
        "Q：上下文工程是怎么设计的？",
        "在项目里，每个 Agent 节点只接收它需要的信息。method_node 只读 methodology 章节的\n"
        "字符范围，不把全文塞进去。具体实现是每个节点返回 {**state, 'new_key': value}，\n"
        "精确控制下游节点的上下文范围，节省 token，减少噪声干扰。"
    ),
    (
        "Q：记忆机制是怎么做的？",
        "两层记忆设计。短期记忆：LangGraph state 在节点间传递，执行完自动清理；\n"
        "长期记忆：eval_results.json 持久化存储每次评测结果（论文名/评分/原因/时间戳）。\n"
        "下次批量运行时读取历史跳过已评测论文，不重复消耗 API 费用。"
    ),
    (
        "Q：Agent 的任务规划是怎么做的？（ReAct、Plan-and-Execute）",
        "两种都有体现。7节点流水线是 Plan-and-Execute：先规划好执行顺序再依次执行，\n"
        "每步结果传递给下一步。低分重试闭环是 ReAct：judge 打分（观察）→\n"
        "生成改进指令（思考）→ 执行模型重做节点（行动）→ 再打分（观察），循环最多两次。"
    ),
    (
        "Q：Modular Agent 多步规划具体怎么做？",
        "执行与评测完全分离：llm_executor（gpt-4o-mini）负责读论文提取信息，\n"
        "llm_judge（gpt-4o）负责评估输出质量。好处一：避免自评偏差；\n"
        "好处二：成本控制，执行用便宜模型，评测用准确模型。\n"
        "每个节点独立，某个技能失败只重做那个节点，不影响其他节点结果。"
    ),
    (
        "Q：如何设计一套评估方案判断 AI 系统好不好？",
        "Skill Rubric 方案：6个维度独立评分（0~1），五档标准（1.0/0.75/0.5/0.25/0.0），\n"
        "不是模糊打分。用更强的模型（gpt-4o）作 judge，防止自评偏差。\n"
        "结果写入 Excel 可横向对比多篇论文。\n"
        "实测：22篇论文，skill4 通过重试从0.25提升到1.00，验证闭环有效。"
    ),
    (
        "Q：如何提升相关度/优化回答效果？",
        "低分触发反馈重试。judge 不只给分，还给出具体改进指令，\n"
        "如'没有列出 baseline 名字，重提取时必须明确写出对比方法名称'。\n"
        "执行模型针对性重做，不是盲目重跑全部。\n"
        "今天实测：skill5 因连接错误得0.00，重试后恢复0.75，容错机制有效。"
    ),
    (
        "Q：高频请求延迟高怎么优化？",
        "arXiv API 触发429限速后，实现指数退避重试：5s → 10s → 20s → 40s。\n"
        "原理与滑动窗口限流一致，检测到限速就主动降低请求频率，不是一直重试雪崩。\n"
        "查询之间固定等待3秒，下载之间等待1秒，从源头控制速率。"
    ),
    (
        "Q：是否使用过 LangChain 等 Agent 框架？",
        "使用 LangGraph（LangChain 生态）构建 StateGraph，7节点有向图，\n"
        "支持 MemorySaver checkpointer 断点续跑。\n"
        "相比 LangChain 的 AgentExecutor，LangGraph 更适合多步骤有状态的 Agent 流程，\n"
        "节点间数据流更清晰，便于调试和局部重试。"
    ),
]

for q, a in qa_list:
    lines.append("")
    lines.append(f"  {q}")
    for aline in a.split("\n"):
        lines.append(f"    {aline}")

# ══ 五、后续计划 ══
h1("五、后续计划（分步骤）")
steps = [
    ("Step 1",  "完成", "跑通完整流程 fetch + evaluate",              "已完成 2026-04-01"),
    ("Step 2",  "完成", "修复 arXiv 搜索：加 cs 分类过滤",            "已完成 2026-04-01"),
    ("Step 3",  "完成", "修复 skill6 Rubric：去掉硬性差值要求",        "已完成 2026-04-01"),
    ("Step 4",  "待做", "论文摘要 embedding 索引 + 语义检索",          "RAG检索、父子索引"),
    ("Step 5",  "待做", "BM25 + 向量检索混合融合",                    "混合检索比例"),
    ("Step 6",  "待做", "rerank 重排序模块",                          "rerank 后返回几个块"),
    ("Step 7",  "待做", "布隆过滤器替换 set 做论文去重",               "布隆过滤器实践"),
    ("Step 8",  "待做", "显式实现令牌桶限流器",                        "令牌桶/漏桶/滑动窗口"),
    ("Step 9",  "待做", "FastAPI 后端，暴露评测数据接口",              "后端设计"),
    ("Step 10", "待做", "SQLite 替换 JSON，加数据库索引",              "数据库/B+树"),
    ("Step 11", "待做", "前端页面 + 用户行为采集",                     "前端交互"),
    ("Step 12", "待做", "推荐引擎（内容相似度 + 行为数据）",           "推荐系统"),
]
lines.append(f"  {'步骤':<10} {'状态':<6} {'内容':<38} {'备注'}")
lines.append("  " + "-" * 72)
for step, status, content, note in steps:
    lines.append(f"  {step:<10} {status:<6} {content:<38} {note}")

# ══ 六、老师评语 ══
h1("六、严格老师评语")
para("""\
【肯定】首次完整运行成功，22篇论文全部跑通，重试机制有效，Excel双语导出正常。
这是从"能跑"到"真的跑起来了"的关键一步，值得肯定。
两个系统性问题在发现当天同步修复，执行力强。

【批评】arXiv 搜索出来的论文里有热力学、心脏建模、物理学，
说明还没有建立"数据质量"意识。跑了22篇，有效的可能只有3~4篇，
其余全是噪声，白白浪费了 API 费用。
skill6 卡在0.50本来应该在设计阶段就发现，测试覆盖不足。

【要求】Step 4（RAG检索）开始之前，先用修复后的搜索再跑一次，
确认无关论文比例大幅下降，再往下推进。质量比数量重要。""")

lines.append("")
lines.append("=" * 60)
lines.append(f"日志生成时间：{now}")
lines.append("=" * 60)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"日志已生成：{output_path}")
