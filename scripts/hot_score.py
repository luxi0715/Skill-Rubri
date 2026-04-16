"""
hot_score.py
-------------
Redis ZSet 热度分管理。

热度公式：
  score = likes×3 + comments×2 + views×1 + quality×10

ZSet key：papers:hot
  member = paper 文件名（如 2022_Xie_CL4SRec.pdf）
  score  = 热度分（浮点）

启动时调用 rebuild_hot_scores() 从 SQLite 全量初始化。
点赞/浏览/评论发生时调用 increment_hot_score() 增量更新。

Redis 不可用时所有操作静默跳过，首页退回 SQL 子查询排序。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

ZSET_KEY = "papers:hot"

# ── Redis 连接（复用 api_server 的连接逻辑）─────────────
def _get_redis():
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0,
                        decode_responses=True, socket_connect_timeout=1)
        r.ping()
        return r
    except Exception:
        return None


# ── 计算单篇热度分 ──────────────────────────────────────
def calc_score(likes: int, comments: int, views: int, quality: float) -> float:
    return likes * 3 + comments * 2 + views * 1 + quality * 10


# ── 全量重建 ZSet（启动时调用）─────────────────────────
def rebuild_hot_scores():
    """
    从 SQLite 读全部论文，重新计算热度分并写入 Redis ZSet。
    """
    r = _get_redis()
    if not r:
        return False

    from db import get_connection
    conn = get_connection()
    papers = conn.execute("SELECT paper, likes, paper_quality_score FROM papers").fetchall()

    mapping = {}
    for p in papers:
        paper   = p["paper"]
        likes   = p["likes"] or 0
        quality = p["paper_quality_score"] or 0

        comments = conn.execute(
            "SELECT COUNT(*) as cnt FROM comments WHERE paper=?", (paper,)
        ).fetchone()["cnt"]

        views = conn.execute(
            "SELECT COUNT(*) as cnt FROM user_behaviors WHERE paper=? AND action='view'",
            (paper,)
        ).fetchone()["cnt"]

        mapping[paper] = calc_score(likes, comments, views, quality)

    conn.close()

    if mapping:
        r.delete(ZSET_KEY)
        r.zadd(ZSET_KEY, mapping)

    return True


# ── 增量更新：点赞 ────────────────────────────────────
def on_like(paper: str):
    r = _get_redis()
    if r:
        r.zincrby(ZSET_KEY, 3, paper)   # 点赞权重 3


# ── 增量更新：浏览 ────────────────────────────────────
def on_view(paper: str):
    r = _get_redis()
    if r:
        r.zincrby(ZSET_KEY, 1, paper)   # 浏览权重 1


# ── 增量更新：新评论 ──────────────────────────────────
def on_comment(paper: str):
    r = _get_redis()
    if r:
        r.zincrby(ZSET_KEY, 2, paper)   # 评论权重 2


# ── 查询热度榜（首页 hot 排序时调用）────────────────
def get_hot_ranking(page: int = 1, size: int = 10) -> list | None:
    """
    返回 [(paper_name, score), ...] 或 None（Redis 不可用时退回 SQL）。
    ZREVRANGEBYSCORE 按分从高到低，配合 LIMIT offset,count 分页。
    """
    r = _get_redis()
    if not r:
        return None

    offset = (page - 1) * size
    items  = r.zrevrange(ZSET_KEY, offset, offset + size - 1, withscores=True)
    return [(paper, score) for paper, score in items] if items else None


# ── 查询单篇热度分 ──────────────────────────────────
def get_paper_hot_score(paper: str) -> float:
    r = _get_redis()
    if not r:
        return 0.0
    score = r.zscore(ZSET_KEY, paper)
    return float(score) if score is not None else 0.0
