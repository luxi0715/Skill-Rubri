"""
api_server.py
--------------
FastAPI 论文检索服务

原理：
  uvicorn 监听 8000 端口，收到 HTTP 请求后交给 FastAPI 路由分发。
  FastAPI 用装饰器 @app.get("/路径") 把函数绑定到 URL，
  函数参数自动从 URL 查询字符串解析，返回值自动序列化成 JSON。

  /search  → 调用 hybrid_search.py 里的检索逻辑（BM25+向量+混合）
  /paper   → 从 eval_results.json 读取单篇详情
  /stats   → 读取 paper_meta.json 做统计

接口文档（自动生成）：
  启动后浏览器打开 http://localhost:8000/docs

右键运行：
  PyCharm 直接右键 Run 'api_server'，或 Ctrl+Shift+F10
"""

import json
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 把 scripts/ 目录加入 path，让 hybrid_search 可以直接 import
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE_DIR, "data")

sys.path.insert(0, SCRIPTS_DIR)

from hybrid_search import (
    encode_api, search_bm25,
    search_vector, hybrid_scores, top_k_results,
    recommend_similar,
)
from db import (init_db, get_all_papers, get_paper,
                like_paper, get_paper_images,
                add_comment, get_comments, like_comment,
                create_user, get_user_by_email, get_user_by_id,
                record_behavior, get_paper_view_count,
                get_papers_page)
from auth import (hash_password, verify_password, create_token,
                  get_current_user, get_optional_user)
from hot_score import on_like, on_view, on_comment, rebuild_hot_scores
from kafka_producer import produce_behavior, produce_hot_delta
from metrics import (
    search_requests_total, search_duration_seconds,
    cache_hits_total, cache_misses_total,
    paper_likes_total, paper_views_total, paper_comments_total,
    faiss_search_duration_seconds,
)
import time as _time

# ── Redis 缓存（可选，不启动 Redis 也能正常运行）──────────
import hashlib
try:
    import redis as _redis
    _redis_client = _redis.Redis(host="localhost", port=6379, db=0,
                                 decode_responses=True, socket_connect_timeout=1)
    _redis_client.ping()
    REDIS_OK = True
    print("  [Redis] 连接成功，搜索缓存已启用")
except Exception:
    _redis_client = None
    REDIS_OK = False
    print("  [Redis] 未连接，搜索缓存跳过（不影响服务）")

SEARCH_CACHE_TTL = 300   # 搜索结果缓存 5 分钟


def _cache_get(key: str):
    if not REDIS_OK:
        return None
    try:
        val = _redis_client.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def _cache_set(key: str, value, ttl: int = SEARCH_CACHE_TTL):
    if not REDIS_OK:
        return
    try:
        _redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception:
        pass


# ── 应用初始化 ──────────────────────────────────────────────
app = FastAPI(
    title="论文检索 API",
    description="BM25 + 向量混合检索，基于 44 篇 AI 推荐系统论文",
    version="1.0.0",
)

# 静态文件服务（论文图片）
IMG_DIR      = os.path.join(BASE_DIR, "data", "images")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(IMG_DIR, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=IMG_DIR), name="images")

# 允许跨域（方便以后接前端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 启动时加载数据（只加载一次）────────────────────────────
_records = None

def clean_field(val):
    """过滤掉评测阶段产生的错误信息，避免暴露给前端"""
    if val and (val.startswith("[ERROR") or val.startswith("[错误")):
        return ""
    return val or ""


def is_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def translate_query(q: str) -> str:
    """中文查询 → 英文，提升 FAISS 检索质量"""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Translate the following search query to English. Output ONLY the translated query, nothing else.\n\n{q}"
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def get_records():
    """返回论文列表，启动后首次调用时加载，后续复用内存缓存。"""
    global _records
    if _records is None:                 # Bug Fix：之前每次都重新查库
        _records = get_all_papers()
    return _records



# ── 接口 1：检索 ────────────────────────────────────────────
@app.get("/search")
def search(
    q:      str = Query(...,        description="搜索关键词，例如：contrastive learning cold start"),
    mode:   str = Query("hybrid",  description="检索模式：bm25 / vector / hybrid"),
    top_k:  int = Query(5,         description="返回结果数量", ge=1, le=20),
    alpha:  float = Query(0.7,     description="hybrid 模式下向量权重（0~1）", ge=0.0, le=1.0),
):
    """
    论文检索接口

    - **bm25**：关键词精确匹配，适合专有名词（如 NeuralUCB、CL4SRec）
    - **vector**：语义检索，适合描述性查询（如"冷启动推荐"）
    - **hybrid**：加权融合，通常效果最好
    """
    _t0 = _time.time()

    # ── Redis 缓存命中检查 ──
    cache_key = "search:" + hashlib.md5(f"{q}:{mode}:{top_k}:{alpha}".encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["cache"] = "hit"
        cache_hits_total.inc()
        search_requests_total.labels(mode=mode, cache="hit").inc()
        search_duration_seconds.labels(mode=mode).observe(_time.time() - _t0)
        return cached

    cache_misses_total.inc()
    records  = get_records()
    search_q = translate_query(q) if is_chinese(q) else q

    if mode == "bm25":
        scores = search_bm25(search_q, records, top_k=top_k)

    elif mode == "vector":
        _tv = _time.time()
        vec    = encode_api(search_q)
        scores = search_vector(vec, top_k=top_k)
        faiss_search_duration_seconds.observe(_time.time() - _tv)

    elif mode == "hybrid":
        bm25_sc = search_bm25(search_q, records, top_k=top_k)
        _tv = _time.time()
        vec     = encode_api(search_q)
        vec_sc  = search_vector(vec, top_k=top_k)
        faiss_search_duration_seconds.observe(_time.time() - _tv)
        scores  = hybrid_scores(bm25_sc, vec_sc, alpha=alpha)

    else:
        raise HTTPException(status_code=400, detail=f"不支持的 mode：{mode}，请用 bm25/vector/hybrid")

    results = top_k_results(scores, top_k=top_k, label=mode)
    results = [r for r in results if r["score"] >= 0.3]

    search_requests_total.labels(mode=mode, cache="miss").inc()
    search_duration_seconds.labels(mode=mode).observe(_time.time() - _t0)

    resp_data = {
        "query":   q,
        "mode":    mode,
        "top_k":   top_k,
        "results": results,
        "cache":   "miss",
    }
    _cache_set(cache_key, resp_data)
    return resp_data


# ── 接口 2：单篇详情 ────────────────────────────────────────
@app.get("/paper/{paper_name}")
def paper_detail(paper_name: str):
    """
    返回单篇论文的完整评测内容

    paper_name 为文件名（不含 .pdf），例如：
    2022_Xie_Contrastive_Learning_for_Sequential_Reco
    """
    r = get_paper(paper_name)
    if r:
        return {
            "paper":               r.get("paper"),
            "paper_quality_score": r.get("paper_quality_score"),
            "skill_rubric_total":  r.get("skill_rubric_total"),
            "research_question":   clean_field(r.get("research_question_cn") or r.get("research_question")),
            "methodology":         clean_field(r.get("methodology_cn")        or r.get("methodology")),
            "datasets":            clean_field(r.get("datasets_cn")           or r.get("datasets")),
            "results":             clean_field(r.get("results_cn")            or r.get("results")),
            "critique":            clean_field(r.get("critique_cn")           or r.get("critique")),
            "comment":             clean_field(r.get("comment_cn")            or r.get("comment")),
            "title_cn":            r.get("title_cn", ""),
            "likes":               r.get("likes", 0),
            "arxiv_id":            r.get("arxiv_id", ""),
            "skill_scores":        r.get("skill_scores", {}),
            "skill_reasons":       r.get("skill_reasons", {}),
        }

    raise HTTPException(status_code=404, detail=f"未找到论文：{paper_name}")


# ── 接口 3：分页论文列表（首页信息流）──────────────────────
@app.get("/papers")
def all_papers(
    sort:   str = Query("hot",  description="排序方式：hot / new / quality"),
    page:   int = Query(1,      description="页码，从1开始", ge=1),
    size:   int = Query(10,     description="每页数量", ge=1, le=50),
    domain: str = Query("",     description="按领域筛选，空字符串表示全部"),
):
    """首页信息流，支持热度/最新/质量分排序 + 分页"""
    result = get_papers_page(sort=sort, page=page, size=size, domain=domain)
    result["papers"] = [
        {
            "paper":               m.get("paper"),
            "paper_quality_score": m.get("paper_quality_score"),
            "skill_rubric_total":  m.get("skill_rubric_total"),
            "comment":             clean_field(m.get("comment_cn") or m.get("comment")),
            "title_cn":            m.get("title_cn", ""),
            "domain":              m.get("domain", ""),
            "year":                m.get("year", ""),
            "venue":               m.get("venue", ""),
            "open_source":         m.get("open_source", ""),
            "likes":               m.get("likes", 0),
            "hot_score":           m.get("hot_score", 0),
        }
        for m in result["papers"]
    ]
    return result


# ── 接口：注册 ─────────────────────────────────────────────
class RegisterBody(BaseModel):
    email:    str
    nickname: str
    password: str

@app.post("/auth/register")
def register(body: RegisterBody):
    if not body.email.strip() or not body.password.strip() or not body.nickname.strip():
        raise HTTPException(status_code=400, detail="邮箱、昵称和密码不能为空")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位")
    if get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="该邮箱已注册")
    try:
        user  = create_user(body.email, body.nickname, hash_password(body.password))
        token = create_token(user["id"], user["email"], user["nickname"])
        return {"token": token, "nickname": user["nickname"], "user_id": user["id"]}
    except Exception:
        raise HTTPException(status_code=500, detail="注册失败，请重试")


# ── 接口：登录 ──────────────────────────────────────────────
class LoginBody(BaseModel):
    email:    str
    password: str

@app.post("/auth/login")
def login(body: LoginBody):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    token = create_token(user["id"], user["email"], user["nickname"])
    return {"token": token, "nickname": user["nickname"], "user_id": user["id"]}


# ── 接口：获取当前用户信息 ───────────────────────────────────
@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    user = get_user_by_id(int(current_user["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ── 接口：浏览记录 ──────────────────────────────────────────
@app.post("/view/{paper_name}")
def record_view(paper_name: str, authorization: str = Header(None)):
    paper = paper_name if paper_name.endswith(".pdf") else paper_name + ".pdf"
    user  = get_optional_user(authorization)
    uid   = int(user["sub"]) if user else None
    sent = produce_behavior(paper, "view", user_id=uid)
    if not sent:             # Kafka 不可用 → 同步写 DB
        record_behavior(paper, "view", user_id=uid)
    on_view(paper)           # Redis ZSet 热度 +1
    paper_views_total.inc()  # Prometheus 指标
    return {"ok": True}


# ── 接口：论文点赞 ──────────────────────────────────────────
@app.post("/like/{paper_name}")
def like(paper_name: str):
    paper = paper_name + ".pdf" if not paper_name.endswith(".pdf") else paper_name
    likes = like_paper(paper)
    on_like(paper)                          # Redis ZSet 热度 +3
    produce_behavior(paper, "like")         # Kafka 异步记录（失败静默）
    paper_likes_total.inc()                 # Prometheus 指标
    return {"likes": likes}


# ── 接口：论文图片 ──────────────────────────────────────────
@app.get("/images/{paper_name}")
def paper_images(paper_name: str):
    paper = paper_name if paper_name.endswith(".pdf") else paper_name + ".pdf"
    imgs, summary = get_paper_images(paper)
    # 把相对路径转成可访问的 URL
    for img in imgs:
        rel = img["image_path"].replace("data/images/", "").replace("\\", "/")
        img["url"] = f"/static/images/{rel}"
    return {"images": imgs, "summary": summary}


# ── 接口：获取评论 ──────────────────────────────────────────
@app.get("/comments/{paper_name}")
def get_paper_comments(paper_name: str):
    paper = paper_name if paper_name.endswith(".pdf") else paper_name + ".pdf"
    return {"comments": get_comments(paper)}


# ── 接口：提交评论 ──────────────────────────────────────────
class CommentBody(BaseModel):
    paper:    str
    nickname: str
    content:  str

@app.post("/comment")
def post_comment(body: CommentBody):
    if not body.nickname.strip() or not body.content.strip():
        raise HTTPException(status_code=400, detail="昵称和内容不能为空")
    if len(body.content) > 500:
        raise HTTPException(status_code=400, detail="评论不超过500字")
    paper = body.paper if body.paper.endswith(".pdf") else body.paper + ".pdf"
    c = add_comment(paper, body.nickname.strip()[:20], body.content.strip())
    on_comment(paper)        # Redis ZSet 热度 +2
    return c


# ── 接口：评论点赞 ──────────────────────────────────────────
@app.post("/comment/{comment_id}/like")
def like_comment_api(comment_id: int):
    likes = like_comment(comment_id)
    return {"likes": likes}


# ── 接口：相似论文推荐 ─────────────────────────────────────
@app.get("/recommend/{paper_name}")
def recommend(paper_name: str, top_k: int = Query(5, ge=1, le=10)):
    """基于内容的相似论文推荐（FAISS最近邻）"""
    records = get_records()
    paper   = paper_name if paper_name.endswith(".pdf") else paper_name + ".pdf"
    results = recommend_similar(paper, records, top_k=top_k)
    return {"paper": paper_name, "recommendations": results}


# ── Prometheus 指标暴露 ─────────────────────────────────────
from fastapi.responses import Response as FastAPIResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics/python")
def prometheus_metrics():
    """供 Prometheus 抓取的指标端点"""
    return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── 前端页面 ────────────────────────────────────────────────
@app.get("/")
def frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── 接口 5：系统统计 ────────────────────────────────────────
@app.get("/stats")
def stats():
    """
    返回系统整体统计信息：论文总数、平均分、skill 各维度分布
    """
    records = get_records()

    quality_scores = [r.get("paper_quality_score", 0) for r in records]
    rubric_totals  = [r.get("skill_rubric_total",  0) for r in records]

    skill_avgs = {}
    for skill in ["skill1", "skill2", "skill3", "skill4", "skill5", "skill6"]:
        vals = [r["skill_scores"][skill] for r in records if r.get("skill_scores", {}).get(skill) is not None]
        skill_avgs[skill] = round(sum(vals) / len(vals), 3) if vals else 0

    return {
        "total_papers":        len(records),
        "avg_quality_score":   round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
        "avg_rubric_total":    round(sum(rubric_totals)  / len(rubric_totals),  3) if rubric_totals  else 0,
        "skill_averages":      skill_avgs,
        "top3_papers": sorted(
            [{"paper": r["paper"], "score": r.get("paper_quality_score", 0)} for r in records],
            key=lambda x: x["score"], reverse=True
        )[:3],
    }


# ── 右键运行入口 ────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 55)
    print("  论文检索 API 服务启动中...")
    print("  文档地址：http://localhost:8000/docs")
    print("  检索示例：http://localhost:8000/search?q=contrastive+learning&mode=hybrid")
    print("=" * 55)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,     # 右键运行时不需要热重载
    )
