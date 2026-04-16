import os
import json
import uuid
import logging
import datetime
import pdfplumber
import openpyxl
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

PROMPT_VERSION = "v4"

# ================== 日志 ==================
logging.basicConfig(
    filename="eval.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ================== 模型：执行/评测分离 ==================
llm_executor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_judge    = ChatOpenAI(model="gpt-4o",      temperature=0)

# ================== Skill Rubric（6个技能，key不重复） ==================
SKILL_RUBRICS = {
    "skill1": (                          # 研究问题提取
        "You are a strict academic reviewer. Score the research question extraction (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Clearly identifies specific problem, background, core challenge, and motivation\n"
        "0.75 - Identifies most elements but missing some details\n"
        "0.5 - Partially identifies the problem, lacks specifics\n"
        "0.25 - Generic, could apply to any paper\n"
        "0.0 - No meaningful research question\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
    "skill2": (                          # 方法论识别
        "You are a strict academic reviewer. Score the methodology identification (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Names specific algorithm, lists core components and loss functions\n"
        "0.75 - Names algorithm and most components, minor gaps\n"
        "0.5 - Names algorithm only, no component details\n"
        "0.25 - Vague, no specific algorithm named\n"
        "0.0 - No meaningful methodology\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
    "skill3": (                          # 数据集识别
        "You are a strict academic reviewer. Score the dataset identification (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Names all datasets with characteristics, split ratios, and evaluation metrics\n"
        "0.75 - Names datasets with partial descriptions\n"
        "0.5 - Names datasets only, no characteristics\n"
        "0.25 - Vague dataset references, no specific names\n"
        "0.0 - No dataset information\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
    "skill4": (                          # 结果提取
        "You are a strict academic reviewer. Score the experimental results extraction (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Specific metric values, baseline names, improvement margins, dataset names all present\n"
        "0.75 - Most elements present, missing one detail\n"
        "0.5 - Some numbers but missing baselines or dataset names\n"
        "0.25 - Vague, no specific numbers\n"
        "0.0 - No meaningful results\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
    "skill5": (                          # 批判性分析
        "You are a strict academic reviewer. Score the critical analysis (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Paper-specific limitations with concrete scenarios and targeted improvements\n"
        "0.75 - Specific limitations but generic improvements\n"
        "0.5 - Mix of specific and generic criticisms\n"
        "0.25 - Mostly generic phrases applicable to any paper\n"
        "0.0 - Entirely generic or meaningless\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
    "skill6": (                          # 综合打分
        "You are a strict academic reviewer. Score the overall scoring output (0.0-1.0).\n\n"
        "Criteria:\n"
        "1.0 - Scores are logically consistent with the analysis content; each score is clearly justified by specific evidence from the paper\n"
        "0.75 - Scores are mostly consistent with the analysis; minor gaps between scores and supporting reasons\n"
        "0.5 - Scores are partially justified; some dimensions lack specific evidence or the reasoning is vague\n"
        "0.25 - Scores are largely inconsistent with the analysis; reasons do not support the given scores\n"
        "0.0 - No meaningful scoring, or scores completely contradict the analysis\n\n"
        "NOTE: Do NOT penalize for clustered scores. Academic papers naturally score similarly across dimensions. "
        "Judge only whether each score is logically supported by the analysis provided.\n\n"
        "Output to score:\n{output}\n\n"
        'Respond ONLY with JSON: {{"score": 0.XX, "reason": "one sentence"}}'
    ),
}

SKILL_NAMES = {
    "skill1": "研究问题提取",
    "skill2": "方法论识别",
    "skill3": "数据集识别",
    "skill4": "结果提取",
    "skill5": "批判性分析",
    "skill6": "综合打分",
}

# ================== 工具函数 ==================
def load_input(source: str) -> str:
    if source.strip().endswith(".pdf") and os.path.isfile(source.strip()):
        print(f"📄 读取 PDF：{source.strip()}")
        text = ""
        with pdfplumber.open(source.strip()) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        print(f"✅ 共 {len(text)} 字符")
        log.info(f"PDF loaded: {len(text)} chars")
        return text
    print("📝 直接文本输入")
    return source


def invoke_with_retry(llm, prompt: str, retries: int = 3) -> str:
    """带重试的模型调用，失败时最多重试3次"""
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            log.warning(f"调用失败 第{attempt}次: {e}")
            if attempt == retries:
                log.error(f"超过最大重试次数，放弃: {e}")
                return f"[ERROR after {retries} retries: {e}]"


def parse_json_response(raw: str) -> dict:
    """从模型输出中提取 JSON，兼容带 markdown 的输出"""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(raw)


# ================== 节点定义 ==================

def structure_parser(state: dict) -> dict:
    """预处理：让模型自己识别各章节的字符范围"""
    content = state["paper"]
    total = len(content)
    prompt = (
        "You are reading an academic paper. Identify the approximate character position ranges "
        "for each major section based on the content below.\n\n"
        "Output ONLY valid JSON:\n"
        '{"introduction": [start, end], "methodology": [start, end], '
        '"experiments": [start, end], "results": [start, end], "conclusion": [start, end]}\n\n'
        f"Total paper length: {total} characters\n\n"
        f"Paper (first 8000 chars):\n{content[:8000]}"
    )
    default_sections = {
        "introduction": [0,              int(total * 0.20)],
        "methodology":  [int(total*0.15),int(total * 0.55)],
        "experiments":  [int(total*0.40),int(total * 0.75)],
        "results":      [int(total*0.55),total],
        "conclusion":   [int(total*0.85),total],
    }
    try:
        raw = invoke_with_retry(llm_executor, prompt)
        sections = parse_json_response(raw)
        for key in sections:
            sections[key][0] = max(0, min(sections[key][0], total))
            sections[key][1] = max(0, min(sections[key][1], total))
        print(f"  ✓ 结构解析完成")
        log.info(f"Sections: {sections}")
    except Exception as e:
        sections = default_sections
        print(f"  ⚠ 结构解析失败，使用默认范围: {e}")
        log.warning(f"structure_parser fallback: {e}")
    # paper 只在这里传一次，后续节点通过 state["paper"] 读取
    return {"paper": content, "sections": sections}


def question_extractor(state: dict) -> dict:
    """技能1：研究问题提取 — 读 introduction"""
    content = state["paper"]
    sections = state.get("sections", {})
    s, e = sections.get("introduction", [0, int(len(content) * 0.20)])
    prompt = (
        "Please complete Skill 1: Research Question Extraction.\n"
        "Identify from the introduction:\n"
        "- Background: limitations of existing methods\n"
        "- Core challenge: why this problem is hard\n"
        "- Motivation: value of solving this problem\n"
        "Be specific to this paper.\n\n"
        f"Introduction:\n{content[s:e]}"
    )
    result = invoke_with_retry(llm_executor, prompt)
    print("  ✓ 技能1：研究问题提取")
    log.info("skill1 done")
    return {**state, "research_question": result}


def method_node(state: dict) -> dict:
    """技能2：方法论识别 — 读 methodology"""
    content = state["paper"]
    sections = state.get("sections", {})
    s, e = sections.get("methodology", [int(len(content)*0.15), int(len(content)*0.55)])
    prompt = (
        "Please complete Skill 2: Methodology Identification.\n"
        "Identify from the methodology section:\n"
        "- Specific algorithm or model name\n"
        "- Core components (modules, loss functions, training strategy)\n"
        "- Key differences from existing methods\n"
        "Name the exact algorithm.\n\n"
        f"Research question context:\n{state.get('research_question','')}\n\n"
        f"Methodology section:\n{content[s:e]}"
    )
    result = invoke_with_retry(llm_executor, prompt)
    print("  ✓ 技能2：方法论识别")
    log.info("skill2 done")
    return {**state, "methodology": result}


def dataset_node(state: dict) -> dict:
    """技能3：数据集识别 — 读 experiments"""
    content = state["paper"]
    sections = state.get("sections", {})
    s, e = sections.get("experiments", [int(len(content)*0.40), int(len(content)*0.75)])
    prompt = (
        "Please complete Skill 3: Dataset Identification.\n"
        "Identify from the experimental setup:\n"
        "- Dataset names (e.g. MovieLens-1M, Tiktok, Amazon)\n"
        "- Dataset characteristics (size, sparsity, domain)\n"
        "- Train/val/test split ratios\n"
        "- Evaluation metrics (e.g. Recall@20, NDCG@20)\n"
        "Name every dataset explicitly.\n\n"
        f"Experiments section:\n{content[s:e]}"
    )
    result = invoke_with_retry(llm_executor, prompt)
    print("  ✓ 技能3：数据集识别")
    log.info("skill3 done")
    return {**state, "datasets": result}


def results_extractor(state: dict) -> dict:
    """技能4：结果提取 — 读 results"""
    content = state["paper"]
    sections = state.get("sections", {})
    s, e = sections.get("results", [int(len(content)*0.55), len(content)])
    prompt = (
        "Please complete Skill 4: Experimental Results Extraction.\n"
        "Extract from the results section:\n"
        "- Specific metric values (e.g. Recall@20=0.0812)\n"
        "- Baseline method names\n"
        "- Improvement margins over best baseline\n"
        "- Which dataset each result is from\n"
        "Must include specific numbers.\n\n"
        f"Methodology:\n{state.get('methodology','')}\n\n"
        f"Datasets:\n{state.get('datasets','')}\n\n"
        f"Results section:\n{content[s:e]}"
    )
    result = invoke_with_retry(llm_executor, prompt)
    print("  ✓ 技能4：结果提取")
    log.info("skill4 done")
    return {**state, "results": result}


def critic(state: dict) -> dict:
    """技能5：批判性分析 — 只看前面节点的输出"""
    prompt = (
        "Please complete Skill 5: Critical Analysis.\n"
        "Based on the analysis below, identify specific limitations:\n"
        "- Each limitation must be specific to this paper's method\n"
        "- Explain what scenario makes this limitation problematic\n"
        "- Provide a targeted improvement direction\n"
        "FORBIDDEN generic phrases: 'lacks theoretical foundation', 'needs more data'\n\n"
        f"Methodology:\n{state.get('methodology','')}\n\n"
        f"Datasets:\n{state.get('datasets','')}\n\n"
        f"Results:\n{state.get('results','')}"
    )
    result = invoke_with_retry(llm_executor, prompt)
    print("  ✓ 技能5：批判性分析")
    log.info("skill5 done")
    return {**state, "critique": result}


def translator(state: dict) -> dict:
    """第8个节点：把5个英文提取字段翻译成中文，存为 _cn 后缀字段"""
    fields = {
        "research_question": state.get("research_question", ""),
        "methodology":       state.get("methodology",       ""),
        "datasets":          state.get("datasets",          ""),
        "results":           state.get("results",           ""),
        "critique":          state.get("critique",          ""),
    }
    cn = translate_to_chinese(fields)
    print("  ✓ 第8节点：翻译为中文完成")
    log.info("translator done")
    return {
        **state,
        "research_question_cn": cn.get("research_question", ""),
        "methodology_cn":       cn.get("methodology",       ""),
        "datasets_cn":          cn.get("datasets",          ""),
        "results_cn":           cn.get("results",           ""),
        "critique_cn":          cn.get("critique",          ""),
    }


def scorer(state: dict) -> dict:
    """技能6：综合打分 — 只看前面节点的输出"""
    prompt = (
        "Please complete Skill 6: Overall Scoring.\n"
        "Based on Skills 1-5 analysis, score the paper.\n\n"
        "Rules:\n"
        "- Scores MUST have differentiation (avoid clustering in 0.8-0.9)\n"
        "- Each score must be logically supported by the analysis\n"
        "- Weak dimensions should score below 0.6\n\n"
        f"Skill 1 - Research Question:\n{state.get('research_question','')}\n\n"
        f"Skill 2 - Methodology:\n{state.get('methodology','')}\n\n"
        f"Skill 3 - Datasets:\n{state.get('datasets','')}\n\n"
        f"Skill 4 - Results:\n{state.get('results','')}\n\n"
        f"Skill 5 - Critique:\n{state.get('critique','')}\n\n"
        "Output ONLY valid JSON:\n"
        '{"overall_score":0.XX,"dimension_scores":{"originality":0.XX,'
        '"methodology":0.XX,"results":0.XX,"clarity":0.XX,"impact":0.XX},'
        '"strengths":"...","weaknesses":"...","suggestions":"..."}'
    )
    raw = invoke_with_retry(llm_executor, prompt)
    try:
        score_dict = parse_json_response(raw)
        print("  ✓ 技能6：综合打分")
        log.info("skill6 done")
        return {**state, "score_result": score_dict}
    except Exception as e:
        log.error(f"scorer JSON parse failed: {e}")
        return {**state, "score_result": {"raw": raw, "parse_error": True}}


# ================== LangGraph（8个节点） ==================
graph_builder = StateGraph(dict)
graph_builder.add_node("structure_parser",   structure_parser)
graph_builder.add_node("question_extractor", question_extractor)
graph_builder.add_node("method_node",        method_node)
graph_builder.add_node("dataset_node",       dataset_node)
graph_builder.add_node("results_extractor",  results_extractor)
graph_builder.add_node("critic",             critic)
graph_builder.add_node("scorer",             scorer)
graph_builder.add_node("translator",         translator)

graph_builder.add_edge(START,              "structure_parser")
graph_builder.add_edge("structure_parser", "question_extractor")
graph_builder.add_edge("question_extractor","method_node")
graph_builder.add_edge("method_node",      "dataset_node")
graph_builder.add_edge("dataset_node",     "results_extractor")
graph_builder.add_edge("results_extractor","critic")
graph_builder.add_edge("critic",           "scorer")
graph_builder.add_edge("scorer",           "translator")
graph_builder.add_edge("translator",       END)

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# ================== 结果存储 ==================
_BASE        = os.path.dirname(os.path.dirname(__file__))   # agenteval_test01/
_DATA        = os.path.join(_BASE, "data")
RESULTS_FILE  = os.path.join(_DATA, "eval_results.json")
EXCEL_FILE_EN = os.path.join(_DATA, "eval_results_en.xlsx")
EXCEL_FILE_CN = os.path.join(_DATA, "eval_results_cn.xlsx")

# 英文 Excel — Sheet1: 论文内容
HEADERS_EN_CONTENT = [
    "Paper Title", "Domain", "Year", "Venue", "Open Source",
    "Research Question", "Methodology", "Datasets", "Results", "Limitations",
    "Paper Quality Score", "Overall Comment",
]
# 英文 Excel — Sheet2: Skill Rubric 评分（Agent行为质量）
HEADERS_EN_RUBRIC = [
    "Paper Title",
    "Skill1 Score", "Skill1 Reason",
    "Skill2 Score", "Skill2 Reason",
    "Skill3 Score", "Skill3 Reason",
    "Skill4 Score", "Skill4 Reason",
    "Skill5 Score", "Skill5 Reason",
    "Skill6 Score", "Skill6 Reason",
    "Rubric Total",
]
# 中文 Excel — 一张表，全部内容
HEADERS_CN = [
    "论文名", "领域", "年份", "会议/期刊", "是否开源",
    "研究问题", "核心方法", "数据集", "实验结果", "局限性",
    "论文质量分", "综合评语",
    "Skill1分", "Skill1原因",
    "Skill2分", "Skill2原因",
    "Skill3分", "Skill3原因",
    "Skill4分", "Skill4原因",
    "Skill5分", "Skill5原因",
    "Skill6分", "Skill6原因",
    "Rubric总分",
]

def extract_meta(paper_text: str) -> dict:
    """从论文开头提取领域/年份/会议/是否开源"""
    prompt = (
        "Extract metadata from this academic paper. Output ONLY valid JSON:\n"
        '{"domain": "...", "year": "...", "venue": "...", "open_source": "yes/no/unknown"}\n\n'
        f"Paper (first 3000 chars):\n{paper_text[:3000]}"
    )
    raw = invoke_with_retry(llm_executor, prompt)
    try:
        return parse_json_response(raw)
    except Exception:
        return {"domain": "", "year": "", "venue": "", "open_source": ""}


def translate_to_chinese(content: dict) -> dict:
    """将英文提取内容翻译为中文"""
    text_block = json.dumps(content, ensure_ascii=False)
    prompt = (
        "Translate the following JSON values into Chinese. "
        "Keep the JSON keys unchanged. Output ONLY valid JSON.\n\n"
        f"{text_block}"
    )
    raw = invoke_with_retry(llm_executor, prompt)
    try:
        return parse_json_response(raw)
    except Exception:
        return content   # 翻译失败则原样返回


def _get_or_create_sheet(wb, title, headers):
    """获取已有 Sheet 或新建并写表头"""
    if title in wb.sheetnames:
        return wb[title]
    ws = wb.create_sheet(title=title)
    ws.append(headers)
    return ws


def _save_wb(wb, path):
    while True:
        try:
            wb.save(path)
            break
        except PermissionError:
            input(f"⚠️  请先关闭 {os.path.basename(path)}，关闭后按回车继续...")


def export_to_excel(result: dict, meta: dict, final_state: dict):
    """
    生成两个独立 Excel 文件：
      eval_results_en.xlsx — 英文，两张表（Paper Content / Skill Rubric）
      eval_results_cn.xlsx — 中文，一张表（全部内容）
    """
    scores  = result["skill_scores"]
    reasons = result["skill_reasons"]

    def clip(text, n=500):
        return str(text)[:n] if text else ""

    content_en = {
        "research_question": clip(final_state.get("research_question", "")),
        "methodology":       clip(final_state.get("methodology",       "")),
        "datasets":          clip(final_state.get("datasets",          "")),
        "results":           clip(final_state.get("results",           "")),
        "critique":          clip(final_state.get("critique",          "")),
    }

    print("🌐 正在翻译为中文...")
    content_cn = translate_to_chinese(content_en)

    title      = result["paper"]
    domain     = meta.get("domain",      "")
    year       = meta.get("year",        "")
    venue      = meta.get("venue",       "")
    open_src   = meta.get("open_source", "")

    # ══════════════════════════════════════════════════
    # 英文 Excel：两张表
    # ══════════════════════════════════════════════════
    wb_en = openpyxl.load_workbook(EXCEL_FILE_EN) if os.path.exists(EXCEL_FILE_EN) \
            else openpyxl.Workbook()
    if not os.path.exists(EXCEL_FILE_EN):
        wb_en.remove(wb_en.active)   # 删除默认空 Sheet

    paper_quality = result.get("paper_quality_score", 0)
    comment       = result.get("comment", "")

    # Sheet1: Paper Content
    ws_content = _get_or_create_sheet(wb_en, "Paper Content", HEADERS_EN_CONTENT)
    ws_content.append([
        title, domain, year, venue, open_src,
        content_en["research_question"],
        content_en["methodology"],
        content_en["datasets"],
        content_en["results"],
        content_en["critique"],
        paper_quality,
        comment,
    ])

    # Sheet2: Skill Rubric
    ws_rubric = _get_or_create_sheet(wb_en, "Skill Rubric", HEADERS_EN_RUBRIC)
    ws_rubric.append([
        title,
        scores.get("skill1", 0), reasons.get("skill1", ""),
        scores.get("skill2", 0), reasons.get("skill2", ""),
        scores.get("skill3", 0), reasons.get("skill3", ""),
        scores.get("skill4", 0), reasons.get("skill4", ""),
        scores.get("skill5", 0), reasons.get("skill5", ""),
        scores.get("skill6", 0), reasons.get("skill6", ""),
        result["skill_rubric_total"],
    ])

    _save_wb(wb_en, EXCEL_FILE_EN)
    print(f"📊 英文 Excel 已导出：{EXCEL_FILE_EN}")

    # ══════════════════════════════════════════════════
    # 中文 Excel：一张表
    # ══════════════════════════════════════════════════
    wb_cn = openpyxl.load_workbook(EXCEL_FILE_CN) if os.path.exists(EXCEL_FILE_CN) \
            else openpyxl.Workbook()
    if not os.path.exists(EXCEL_FILE_CN):
        wb_cn.remove(wb_cn.active)

    ws_cn = _get_or_create_sheet(wb_cn, "论文评测", HEADERS_CN)
    ws_cn.append([
        title, domain, year, venue, open_src,
        content_cn.get("research_question", ""),
        content_cn.get("methodology",       ""),
        content_cn.get("datasets",          ""),
        content_cn.get("results",           ""),
        content_cn.get("critique",          ""),
        paper_quality,
        comment,
        scores.get("skill1", 0), reasons.get("skill1", ""),
        scores.get("skill2", 0), reasons.get("skill2", ""),
        scores.get("skill3", 0), reasons.get("skill3", ""),
        scores.get("skill4", 0), reasons.get("skill4", ""),
        scores.get("skill5", 0), reasons.get("skill5", ""),
        scores.get("skill6", 0), reasons.get("skill6", ""),
        result["skill_rubric_total"],
    ])

    _save_wb(wb_cn, EXCEL_FILE_CN)
    print(f"📊 中文 Excel 已导出：{EXCEL_FILE_CN}")
    log.info("Excel exported: EN + CN")



def save_result(result: dict):
    # 写 JSON（备份）
    history = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []
    history.append(result)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存：{RESULTS_FILE}")

    # 写 SQLite（主数据源）
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from db import init_db, upsert_paper
        init_db()
        upsert_paper(result)
        print(f"🗄  SQLite 已更新：{result.get('paper', '')}")
    except Exception as e:
        print(f"⚠️  SQLite 写入失败（不影响 JSON）：{e}")

    log.info(f"Result saved: {result['thread_id']}")


# ================== Skill Rubric 评测（含低分重试 + 综合评语）==================

def _score_skills(state: dict, target_keys: list) -> tuple:
    """对指定技能列表打分，返回 (scores_dict, reasons_dict)"""
    skill_outputs = {
        "skill1": state.get("research_question", ""),
        "skill2": state.get("methodology",       ""),
        "skill3": state.get("datasets",          ""),
        "skill4": state.get("results",           ""),
        "skill5": state.get("critique",          ""),
        "skill6": json.dumps(state.get("score_result", {}), ensure_ascii=False),
    }
    scores, reasons = {}, {}
    for key in target_keys:
        prompt = SKILL_RUBRICS[key].format(output=skill_outputs[key])
        raw = invoke_with_retry(llm_judge, prompt)
        try:
            parsed = parse_json_response(raw)
            s      = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except Exception as e:
            s, reason = 0.0, f"parse error: {e}"
            log.warning(f"{key} rubric parse error: {e}")
        scores[key]  = s
        reasons[key] = reason
        print(f"  {key}（{SKILL_NAMES[key]}）：{s:.2f} / 1.00  → {reason}")
    return scores, reasons


def _generate_feedback(low_skills: list, scores: dict, reasons: dict) -> dict:
    """为每个低分技能生成具体改进指令"""
    feedback = {}
    for key in low_skills:
        prompt = (
            f"An AI agent scored {scores[key]:.2f}/1.0 on {SKILL_NAMES[key]}.\n"
            f"Judge reason: {reasons[key]}\n\n"
            "Write 1-2 sentences telling the agent exactly what to fix. Be concrete."
        )
        feedback[key] = invoke_with_retry(llm_judge, prompt)
        print(f"  💬 {key} 改进指令：{feedback[key][:80]}...")
    return feedback


def _rerun_skill(skill_key: str, state: dict, feedback: str) -> dict:
    """携带反馈重新执行某个技能节点，返回更新后的 state"""
    content  = state["paper"]
    sections = state.get("sections", {})
    fb = f"\n\n[Feedback from judge]: {feedback}\nPlease fix this specifically.\n"

    if skill_key == "skill1":
        s, e = sections.get("introduction", [0, int(len(content) * 0.20)])
        result = invoke_with_retry(llm_executor,
            "Redo Skill 1: Research Question Extraction." + fb +
            f"\nIntroduction:\n{content[s:e]}")
        return {**state, "research_question": result}

    elif skill_key == "skill2":
        s, e = sections.get("methodology", [int(len(content)*0.15), int(len(content)*0.55)])
        result = invoke_with_retry(llm_executor,
            "Redo Skill 2: Methodology Identification." + fb +
            f"\nResearch question:\n{state.get('research_question','')}\n\n"
            f"Methodology section:\n{content[s:e]}")
        return {**state, "methodology": result}

    elif skill_key == "skill3":
        s, e = sections.get("experiments", [int(len(content)*0.40), int(len(content)*0.75)])
        result = invoke_with_retry(llm_executor,
            "Redo Skill 3: Dataset Identification." + fb +
            f"\nExperiments section:\n{content[s:e]}")
        return {**state, "datasets": result}

    elif skill_key == "skill4":
        s, e = sections.get("results", [int(len(content)*0.55), len(content)])
        result = invoke_with_retry(llm_executor,
            "Redo Skill 4: Experimental Results Extraction." + fb +
            f"\nMethodology:\n{state.get('methodology','')}\n\n"
            f"Datasets:\n{state.get('datasets','')}\n\n"
            f"Results section:\n{content[s:e]}")
        return {**state, "results": result}

    elif skill_key == "skill5":
        result = invoke_with_retry(llm_executor,
            "Redo Skill 5: Critical Analysis." + fb +
            f"\nMethodology:\n{state.get('methodology','')}\n\n"
            f"Datasets:\n{state.get('datasets','')}\n\n"
            f"Results:\n{state.get('results','')}")
        return {**state, "critique": result}

    elif skill_key == "skill6":
        raw = invoke_with_retry(llm_executor,
            "Redo Skill 6: Overall Scoring." + fb +
            f"\nSkill1:\n{state.get('research_question','')}\n\n"
            f"Skill2:\n{state.get('methodology','')}\n\n"
            f"Skill3:\n{state.get('datasets','')}\n\n"
            f"Skill4:\n{state.get('results','')}\n\n"
            f"Skill5:\n{state.get('critique','')}\n\n"
            'Output ONLY valid JSON: {"overall_score":0.XX,"dimension_scores":{'
            '"originality":0.XX,"methodology":0.XX,"results":0.XX,'
            '"clarity":0.XX,"impact":0.XX},"strengths":"...","weaknesses":"...","suggestions":"..."}')
        try:
            return {**state, "score_result": parse_json_response(raw)}
        except Exception:
            return state

    return state


def _generate_overall_comment(state: dict, scores: dict, reasons: dict) -> str:
    """生成 3-5 句综合评语"""
    score_summary = "\n".join([
        f"{SKILL_NAMES[k]}: {scores[k]:.2f} — {reasons[k]}"
        for k in ["skill1","skill2","skill3","skill4","skill5","skill6"]
    ])
    paper_score = state.get("score_result", {})
    prompt = (
        "Based on the skill evaluation below, write a 3-5 sentence overall academic assessment.\n"
        "Be specific to this paper. Mention concrete strengths and weaknesses.\n"
        "End with a recommendation (e.g. 'Recommended for X' or 'Suitable as reference for Y').\n\n"
        f"Skill Scores:\n{score_summary}\n\n"
        f"Paper Quality Score: {json.dumps(paper_score, ensure_ascii=False)}"
    )
    return invoke_with_retry(llm_judge, prompt)


def run_skill_rubric(final_state: dict, max_retries: int = 2) -> dict:
    """Skill Rubric 打分 → 低分重试（最多2次）→ 生成综合评语"""
    ALL_SKILLS = ["skill1","skill2","skill3","skill4","skill5","skill6"]
    state = final_state

    print()
    scores, reasons = _score_skills(state, ALL_SKILLS)

    for attempt in range(1, max_retries + 1):
        low = [k for k, v in scores.items() if v < 0.6]
        if not low:
            break
        print(f"\n  🔄 第{attempt}次重试，低分技能：{[SKILL_NAMES[k] for k in low]}")
        feedbacks = _generate_feedback(low, scores, reasons)

        for key in low:
            state = _rerun_skill(key, state, feedbacks[key])
            print(f"    ✓ {SKILL_NAMES[key]} 已重做")

        print(f"\n  📊 重新评分（第{attempt}次）：")
        new_s, new_r = _score_skills(state, low)
        scores.update(new_s)
        reasons.update(new_r)

    total = sum(scores.values()) / len(scores)
    print(f"\n  ⭐ Skill Rubric 总分（Agent行为质量）：{total:.2f} / 1.00")

    print("\n  ✍️  生成综合评语...")
    comment = _generate_overall_comment(state, scores, reasons)
    print(f"  💬 {comment[:120]}...")

    log.info(f"Rubric total: {total:.2f}")
    return {
        "scores":      scores,
        "reasons":     reasons,
        "total":       round(total, 2),
        "comment":     comment,
        "final_state": state,   # 可能经过重做，用最新 state
    }


# ================== 主函数 ==================
def evaluate(source: str, thread_id: str = None):
    if thread_id is None:
        thread_id = f"eval_{uuid.uuid4().hex[:8]}"
    print(f"\n🔑 评测 ID：{thread_id}  (prompt {PROMPT_VERSION})")
    log.info(f"Start evaluation: {thread_id}")

    paper_text  = load_input(source)
    paper_name  = os.path.basename(source) if source.endswith(".pdf") else "direct_input"
    config      = {"configurable": {"thread_id": thread_id}}

    print("🔎 提取论文元信息...")
    meta = extract_meta(paper_text)

    # paper 只在 input_state 里传一次，节点通过 state["paper"] 读取
    input_state = {"paper": paper_text}

    print("\n🔄 Agent 开始执行7个步骤...")
    try:
        final_state = graph.invoke(input_state, config)
    except Exception as e:
        log.error(f"Graph invoke failed: {e}")
        print(f"❌ Agent 运行失败：{e}")
        return None

    print("\n📊 【技能6】综合打分结果：")
    paper_score = final_state.get("score_result", {})
    print(json.dumps(paper_score, ensure_ascii=False, indent=2))

    print("\n🔍 Skill Rubric 逐项评测（评测模型：gpt-4o）...")
    rubric_result = run_skill_rubric(final_state)
    final_state   = rubric_result["final_state"]   # 使用可能经过重做的最新 state

    # 翻译标题和综合评语为中文
    en_title   = " ".join(paper_name.replace(".pdf","").split("_")[2:]) if paper_name.count("_") >= 2 else paper_name.replace(".pdf","").replace("_"," ")
    en_comment = rubric_result["comment"]
    try:
        title_cn   = llm_executor.invoke(f"Translate this paper title to concise Chinese, keep proper nouns in English if needed. Output ONLY the translation:\n{en_title}").content.strip()
        comment_cn = llm_executor.invoke(f"Translate the following academic paper review to Chinese. Output ONLY the translation:\n{en_comment}").content.strip()
    except Exception:
        title_cn   = ""
        comment_cn = ""

    result = {
        "thread_id":           thread_id,
        "timestamp":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_version":      PROMPT_VERSION,
        "paper":               paper_name,
        "paper_score":         paper_score,
        "paper_quality_score": paper_score.get("overall_score", 0),
        "skill_scores":        rubric_result["scores"],
        "skill_reasons":       rubric_result["reasons"],
        "skill_rubric_total":  rubric_result["total"],
        "comment":             en_comment,
        "title_cn":            title_cn,
        "comment_cn":          comment_cn,
        # 提取内容（英文，供 RAG 索引使用）
        "research_question":   final_state.get("research_question", ""),
        "methodology":         final_state.get("methodology",        ""),
        "datasets":            final_state.get("datasets",           ""),
        "results":             final_state.get("results",            ""),
        "critique":            final_state.get("critique",           ""),
        # 中文版（供前端显示使用）
        "research_question_cn": final_state.get("research_question_cn", ""),
        "methodology_cn":       final_state.get("methodology_cn",       ""),
        "datasets_cn":          final_state.get("datasets_cn",          ""),
        "results_cn":           final_state.get("results_cn",           ""),
        "critique_cn":          final_state.get("critique_cn",          ""),
    }
    save_result(result)
    export_to_excel(result, meta, final_state)
    return result


# ================== 排序 ==================
def sort_excel_by_score():
    """两个 Excel 按论文质量分从高到低重新排列"""
    tasks = [
        (EXCEL_FILE_EN, "Paper Content", "Paper Quality Score"),
        (EXCEL_FILE_CN, "论文评测",      "论文质量分"),
    ]
    for filepath, sheet_name, score_col in tasks:
        if not os.path.exists(filepath):
            continue
        wb = openpyxl.load_workbook(filepath)
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]

        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            continue

        header = list(rows[0])
        data   = [list(r) for r in rows[1:]]

        try:
            score_idx = header.index(score_col)
        except ValueError:
            continue

        data.sort(key=lambda r: float(r[score_idx] or 0), reverse=True)

        # 清空并重写
        ws.delete_rows(1, ws.max_row)
        ws.append(header)
        for row in data:
            ws.append(row)

        _save_wb(wb, filepath)
        print(f"🏆 已排序（高→低）：{os.path.basename(filepath)}  [{sheet_name}]")


# ================== 批量评测 ==================
def evaluate_folder(folder_path: str):
    """扫描文件夹内所有 PDF，跳过已评测，全部跑完后排序"""
    pdfs = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])

    # 读取已评测记录
    already_done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            try:
                already_done = {r["paper"] for r in json.load(f)}
            except Exception:
                pass

    todo = [p for p in pdfs if p not in already_done]

    print(f"\n📂 文件夹：{folder_path}")
    print(f"📄 共 {len(pdfs)} 篇 PDF  |  已评测 {len(already_done)} 篇  |  待评测 {len(todo)} 篇\n")

    if not todo:
        print("✅ 所有 PDF 均已评测，无需重复运行。")
    else:
        for i, pdf in enumerate(todo, 1):
            print(f"\n{'='*55}")
            print(f"[{i}/{len(todo)}]  {pdf}")
            evaluate(os.path.join(folder_path, pdf))

        print(f"\n✅ 批量评测完成，共处理 {len(todo)} 篇")

    print("\n📊 按评分排序 Excel...")
    sort_excel_by_score()


# ================== 入口 ==================
if __name__ == "__main__":
    # 单篇
    # evaluate(r"D:\Skill Rubri\agenteval_test01\Contrastive Learning for Cold-Start Recommendation.pdf")

    # 批量（把所有 PDF 放进同一个文件夹，改成对应路径）
    evaluate_folder(os.path.join(_BASE, "papers"))
