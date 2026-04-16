"""
generate_arch.py
用 Courier New 等宽字体把纯 ASCII 架构图渲染成 PNG
  arch_overview.png  -- 总架构图
  arch_detail.png    -- 分模块详细图（输入/处理/输出）
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


def render_ascii(text, filename, bg="#0D1117", fg="#C9D1D9", fontsize=9, dpi=150):
    lines   = text.split("\n")
    n_lines = len(lines)
    n_chars = max(len(l) for l in lines)
    fig_w   = n_chars * 0.073 + 0.4
    fig_h   = n_lines * 0.158 + 0.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.01, 0.99, text,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=fontsize, fontfamily="Courier New",
            color=fg, linespacing=1.38)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=bg, edgecolor="none")
    plt.close(fig)
    print(f"Generated: {path}")


# ============================================================
#  OVERVIEW
# ============================================================
OVERVIEW = """\
+==============================================================================+
|       AI Paper Evaluation + Retrieval System  --  Overview                   |
+==============================================================================+

  +--------------------------------------------------------------------------+
  |  CONFIG LAYER   topics_config.json                                       |
  |  domain names . trending_count=20 . classic_count=20 . download_dir     |
  +-------------------------------------+------------------------------------+
                                        | enabled domains
                                        v
  +--------------------------------------------------------------------------+
  |  MODULE 1   fetch_papers.py   FETCH LAYER                                |
  |                                                                          |
  |  arXiv API search                                                        |
  |    query + (cat:cs.IR OR cs.LG OR cs.AI OR cs.CV OR cs.CL)             |
  |    trending: last 6 months          classic: 2020~2022                   |
  |  dedup (seen_ids set)  ->  download PDF  ->  skip if exists             |
  |  rate limit: exponential backoff   5s -> 10s -> 20s -> 40s             |
  |  naming: {year}_{first_author}_{title_short}.pdf                         |
  +-------------------------------------+------------------------------------+
                                        | papers/*.pdf  (34 files)
                                        v
  +--------------------------------------------------------------------------+
  |  MODULE 2   paper_rubric_eval.py   EVALUATION LAYER                      |
  |                                                                          |
  |  +-- LangGraph 7-Node Pipeline (Plan-and-Execute) -------------------+  |
  |  |                                                                    |  |
  |  |  [1] structure_parser    -> split PDF into sections               |  |
  |  |  [2] question_extractor  -> research_question                     |  |
  |  |  [3] method_node         -> methodology                           |  |
  |  |  [4] dataset_node        -> datasets                              |  |
  |  |  [5] results_extractor   -> results                               |  |
  |  |  [6] critic              -> critique                              |  |
  |  |  [7] scorer              -> score_result / overall_score          |  |
  |  |                                                                    |  |
  |  |  executor: gpt-4o-mini              judge: gpt-4o                 |  |
  |  +--------------------------------------------------------------------+  |
  |                                                                          |
  |  +-- Skill Rubric + Retry Loop (ReAct) -------------------------------+  |
  |  |  skill1 Research Question . skill2 Methodology . skill3 Datasets   |  |
  |  |  skill4 Results           . skill5 Critical     . skill6 Overall   |  |
  |  |  score < 0.6 -> generate feedback -> redo node -> re-score         |  |
  |  |  max 2 retries per skill                                           |  |
  |  +--------------------------------------------------------------------+  |
  +---+-------------------+------------------+-------------------+----------+
      |                   |                  |                   |
      v                   v                  v                   v
  eval_results.json  eval_results_en    eval_results_cn      eval.log
  . paper name       . Paper Content    . all-in-one CN      . run log
  . paper_score      . Skill Rubric     . 1 sheet
  . skill_scores     . 2 sheets
  . comment
  . research_question  <- used by INDEX LAYER
  . methodology / datasets / results / critique
      |
      v
  +--------------------------------------------------------------------------+
  |  MODULE 3   build_index.py   INDEX LAYER                                 |
  |                                                                          |
  |  concat:  title + research_question + methodology + datasets             |
  |           + results + comment                                            |
  |                                                                          |
  |  Mode A (API)    OpenAI text-embedding-3-small  -> 1536-dim vector      |
  |  Mode B (local)  sentence-transformers MiniLM   ->  384-dim vector      |
  |                                                                          |
  |  normalize  ->  FAISS IndexFlatIP  (inner product = cosine similarity)  |
  +------------------+--------------------------------+----------------------+
                     |                                |
                     v                                v
          paper_index_api.faiss              paper_meta.json
          44 vectors . 1536-dim              paper name . scores . comment
                     |                                |
                     +--------------+-----------------+
                                    |
                                    v
  +--------------------------------------------------------------------------+
  |  MODULE 4   search_papers.py   SEARCH LAYER                              |
  |                                                                          |
  |  input: query string (natural language)                                  |
  |  query -> embedding vector -> FAISS cosine search -> top-k indices      |
  |  indices -> paper_meta.json -> paper name + similarity + scores         |
  |  output: rank . paper . similarity . quality_score . rubric . comment   |
  +--------------------------------------------------------------------------+
                                    |
                                    v
  +----------================================================================+
  |  ROADMAP  (not yet built)                                                |
  |                                                                          |
  |  Step 5   BM25 + Vector hybrid retrieval  (a*vector + b*BM25)           |
  |  Step 6   Rerank module  (top-20 candidates -> top-3 final)             |
  |  Step 7   Bloom filter for paper dedup                                   |
  |  Step 8   Token bucket rate limiter                                      |
  |  Step 9   FastAPI backend  GET /papers  GET /search  POST /behavior     |
  |  Step 10  SQLite replace JSON  +  B-tree index                          |
  |  Step 11  Frontend  +  user behavior tracking                           |
  |  Step 12  Recommendation engine  (content similarity -> behavior data)  |
  +--------------------------------------------------------------------------+
"""

# ============================================================
#  DETAIL
# ============================================================
DETAIL = """\
+==============================================================================+
|       AI Paper Evaluation + Retrieval System  --  Module Detail              |
+==============================================================================+

+----------------------------------------------------------------------------+
|  MODULE 1   fetch_papers.py                                                |
+---------------------+------------------------------+----------------------+
|  INPUT              |  PROCESS                     |  OUTPUT              |
|                     |                              |                      |
|  topics_config.json |  (1) arXiv API query         |  papers/*.pdf        |
|  . domain names     |      all:{query} AND         |  . {year}_{author}   |
|  . trending_count   |      (cat:cs.IR OR cs.LG     |    _{title}.pdf      |
|  . classic_count    |       OR cs.AI OR cs.CV)     |                      |
|                     |      AND submittedDate:[...] |  console progress    |
|                     |                              |  . downloaded count  |
|                     |  (2) dedup                   |  . skipped count     |
|                     |      seen_ids = set()        |                      |
|                     |      skip if id in set       |                      |
|                     |                              |                      |
|                     |  (3) download_pdf()          |                      |
|                     |      skip if file exists     |                      |
|                     |      429 -> backoff 5/10/20s |                      |
|                     |                              |                      |
|                     |  (4) call evaluate_folder()  |                      |
|                     |      trigger Module 2        |                      |
+---------------------+------------------------------+----------------------+

+----------------------------------------------------------------------------+
|  MODULE 2   paper_rubric_eval.py                                           |
+---------------------+------------------------------+----------------------+
|  INPUT              |  PROCESS                     |  OUTPUT              |
|                     |                              |                      |
|  papers/*.pdf       |  (1) pdfplumber extract text |  eval_results.json   |
|                     |                              |  . research_question |
|  .env               |  (2) LangGraph 7-node graph  |  . methodology       |
|  . OPENAI_API_KEY   |      state={**state, k:v}    |  . datasets          |
|                     |      nodes:                  |  . results/critique  |
|  eval_results.json  |      structure_parser        |  . paper_score       |
|  . already_done set |      question_extractor      |  . skill_scores      |
|  . skip if exists   |      method_node             |  . comment           |
|                     |      dataset_node            |                      |
|                     |      results_extractor       |  eval_results_en     |
|                     |      critic / scorer         |  . Paper Content     |
|                     |                              |  . Skill Rubric      |
|                     |  (3) Skill Rubric            |                      |
|                     |      judge: gpt-4o           |  eval_results_cn     |
|                     |      executor: gpt-4o-mini   |  . all-in-one CN     |
|                     |      6 skills  scored 0~1    |                      |
|                     |                              |  eval.log            |
|                     |  (4) retry loop (ReAct)      |                      |
|                     |      score<0.6 -> feedback   |                      |
|                     |      -> redo -> re-score     |                      |
|                     |      max 2 retries           |                      |
|                     |                              |                      |
|                     |  (5) generate comment        |                      |
|                     |      translate CN            |                      |
|                     |      export Excel            |                      |
+---------------------+------------------------------+----------------------+

+----------------------------------------------------------------------------+
|  MODULE 3   build_index.py                                                 |
+---------------------+------------------------------+----------------------+
|  INPUT              |  PROCESS                     |  OUTPUT              |
|                     |                              |                      |
|  eval_results.json  |  (1) filter valid records    |  paper_index_api     |
|  . research_question|      skip if no rq field     |  .faiss              |
|  . methodology      |                              |  . 44 vectors        |
|  . datasets         |  (2) concat text             |  . 1536-dim (API)    |
|  . results          |      title + question        |                      |
|  . comment          |      + method + datasets     |  paper_meta.json     |
|                     |      + results + comment     |  . paper name        |
|  OPENAI_API_KEY     |                              |  . quality score     |
|  or torch (local)   |  (3) embed                   |  . rubric total      |
|                     |      API: embedding-3-small  |  . comment           |
|                     |          batch=20, 1536-dim  |  . timestamp         |
|                     |      local: MiniLM, 384-dim  |                      |
|                     |            GPU/CPU auto      |                      |
|                     |                              |                      |
|                     |  (4) normalize + FAISS       |                      |
|                     |      IndexFlatIP             |                      |
|                     |      inner product = cosine  |                      |
+---------------------+------------------------------+----------------------+

+----------------------------------------------------------------------------+
|  MODULE 4   search_papers.py                                               |
+---------------------+------------------------------+----------------------+
|  INPUT              |  PROCESS                     |  OUTPUT              |
|                     |                              |                      |
|  query string       |  (1) encode query            |  top-k results       |
|  e.g. "cold start"  |      API:   OpenAI embed     |                      |
|                     |      local: MiniLM encode    |  . rank              |
|  paper_index.faiss  |                              |  . paper name        |
|  . 44 vectors       |  (2) FAISS search            |  . similarity score  |
|                     |      IndexFlatIP.search()    |  . quality score     |
|  paper_meta.json    |      returns (scores, idx)   |  . rubric total      |
|  . name/score/      |                              |  . comment excerpt   |
|    comment          |  (3) map index -> meta       |                      |
|                     |      idx[i] -> paper_meta    |                      |
|  mode: api / local  |      [idx[i]]                |                      |
|  top_k: 3 (default) |                              |                      |
|                     |  (4) format & print          |                      |
+---------------------+------------------------------+----------------------+
"""

render_ascii(OVERVIEW, "arch_overview.png", fontsize=8.5)
render_ascii(DETAIL,   "arch_detail.png",   fontsize=8.5)
