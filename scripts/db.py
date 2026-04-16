"""
db.py
------
SQLite 数据库模块，替代 eval_results.json。

提供：
  init_db()         — 建表（首次运行）
  upsert_paper()    — 插入或更新一篇论文
  get_all_papers()  — 返回全部记录（list[dict]）
  get_paper()       — 按论文名查单篇
"""

import json
import os
import sqlite3
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "papers.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper               TEXT PRIMARY KEY,
        thread_id           TEXT,
        timestamp           TEXT,
        prompt_version      TEXT,
        paper_score         REAL,
        paper_quality_score REAL,
        skill_rubric_total  REAL,
        skill_scores        TEXT,
        skill_reasons       TEXT,
        comment             TEXT,
        comment_cn          TEXT,
        title_cn            TEXT,
        research_question   TEXT,
        methodology         TEXT,
        datasets            TEXT,
        results             TEXT,
        critique            TEXT,
        research_question_cn TEXT,
        methodology_cn      TEXT,
        datasets_cn         TEXT,
        results_cn          TEXT,
        critique_cn         TEXT,
        domain              TEXT,
        year                TEXT,
        venue               TEXT,
        open_source         TEXT,
        likes               INTEGER DEFAULT 0,
        arxiv_id            TEXT,
        image_summary       TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS paper_images (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        paper      TEXT NOT NULL,
        image_path TEXT NOT NULL,
        caption    TEXT,
        order_idx  INTEGER DEFAULT 0
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        paper      TEXT NOT NULL,
        nickname   TEXT NOT NULL,
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL,
        likes      INTEGER DEFAULT 0,
        user_id    INTEGER DEFAULT NULL
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        email         TEXT NOT NULL UNIQUE,
        nickname      TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        is_active     INTEGER DEFAULT 1
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS user_behaviors (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER,
        anon_id    TEXT,
        paper      TEXT NOT NULL,
        action     TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    conn.commit()

    # 自动补齐旧数据库缺少的列（ALTER TABLE 忽略已存在的列错误）
    new_columns = [
        ("likes",         "INTEGER DEFAULT 0"),
        ("arxiv_id",      "TEXT DEFAULT ''"),
        ("image_summary", "TEXT DEFAULT ''"),
        ("title_cn",      "TEXT DEFAULT ''"),
        ("comment_cn",    "TEXT DEFAULT ''"),
        ("abstract",      "TEXT DEFAULT ''"),   # 原始摘要（Fix P2/P5）
    ]
    for col, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {col_def}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # 列已存在，忽略

    conn.close()


# ── 点赞 ──────────────────────────────────────────────────
def like_paper(paper: str) -> int:
    conn = get_connection()
    conn.execute("UPDATE papers SET likes = likes + 1 WHERE paper = ?", (paper,))
    conn.commit()
    row = conn.execute("SELECT likes FROM papers WHERE paper = ?", (paper,)).fetchone()
    conn.close()
    return row["likes"] if row else 0


# ── 图片 ──────────────────────────────────────────────────
def save_paper_images(paper: str, images: list, summary: str):
    """images: [{"image_path": ..., "caption": ..., "order_idx": ...}]"""
    conn = get_connection()
    conn.execute("DELETE FROM paper_images WHERE paper = ?", (paper,))
    for img in images:
        conn.execute(
            "INSERT INTO paper_images (paper, image_path, caption, order_idx) VALUES (?,?,?,?)",
            (paper, img["image_path"], img.get("caption", ""), img.get("order_idx", 0))
        )
    conn.execute("UPDATE papers SET image_summary = ? WHERE paper = ?", (summary, paper))
    conn.commit()
    conn.close()


def get_paper_images(paper: str) -> tuple:
    """返回 (images_list, summary_str)"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT image_path, caption, order_idx FROM paper_images WHERE paper = ? ORDER BY order_idx",
        (paper,)
    ).fetchall()
    row = conn.execute("SELECT image_summary FROM papers WHERE paper = ?", (paper,)).fetchone()
    conn.close()
    images  = [dict(r) for r in rows]
    summary = row["image_summary"] if row and row["image_summary"] else ""
    return images, summary


# ── 评论 ──────────────────────────────────────────────────
def add_comment(paper: str, nickname: str, content: str) -> dict:
    from datetime import datetime
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO comments (paper, nickname, content, created_at, likes) VALUES (?,?,?,?,0)",
        (paper, nickname, content, created_at)
    )
    cid = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": cid, "paper": paper, "nickname": nickname,
            "content": content, "created_at": created_at, "likes": 0}


def get_comments(paper: str) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, nickname, content, created_at, likes FROM comments WHERE paper = ? ORDER BY id DESC",
        (paper,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def like_comment(comment_id: int) -> int:
    conn = get_connection()
    conn.execute("UPDATE comments SET likes = likes + 1 WHERE id = ?", (comment_id,))
    conn.commit()
    row = conn.execute("SELECT likes FROM comments WHERE id = ?", (comment_id,)).fetchone()
    conn.close()
    return row["likes"] if row else 0


# ── 用户 ──────────────────────────────────────────────────
def create_user(email: str, nickname: str, password_hash: str) -> dict:
    from datetime import datetime
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, nickname, password_hash, created_at) VALUES (?,?,?,?)",
            (email.lower().strip(), nickname.strip(), password_hash, created_at)
        )
        uid = cur.lastrowid
        conn.commit()
        return {"id": uid, "email": email, "nickname": nickname, "created_at": created_at}
    except Exception as e:
        raise e
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute(
        "SELECT id, email, nickname, password_hash, created_at FROM users WHERE email = ? AND is_active = 1",
        (email.lower().strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute(
        "SELECT id, email, nickname, created_at FROM users WHERE id = ? AND is_active = 1",
        (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ── 行为记录 ───────────────────────────────────────────────
def record_behavior(paper: str, action: str, user_id: int = None, anon_id: str = None):
    """记录用户行为：view / like / search"""
    from datetime import datetime
    conn = get_connection()
    conn.execute(
        "INSERT INTO user_behaviors (user_id, anon_id, paper, action, created_at) VALUES (?,?,?,?,?)",
        (user_id, anon_id, paper, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_paper_view_count(paper: str) -> int:
    conn = get_connection()
    row  = conn.execute(
        "SELECT COUNT(*) as cnt FROM user_behaviors WHERE paper = ? AND action = 'view'",
        (paper,)
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def _to_float(val):
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def upsert_paper(record: dict):
    """插入或更新一篇论文（以 paper 文件名为主键）"""
    conn = get_connection()
    conn.execute("""
    INSERT INTO papers (
        paper, thread_id, timestamp, prompt_version,
        paper_score, paper_quality_score, skill_rubric_total,
        skill_scores, skill_reasons,
        comment, comment_cn, title_cn,
        research_question, methodology, datasets, results, critique,
        research_question_cn, methodology_cn, datasets_cn, results_cn, critique_cn,
        domain, year, venue, open_source, abstract
    ) VALUES (
        :paper, :thread_id, :timestamp, :prompt_version,
        :paper_score, :paper_quality_score, :skill_rubric_total,
        :skill_scores, :skill_reasons,
        :comment, :comment_cn, :title_cn,
        :research_question, :methodology, :datasets, :results, :critique,
        :research_question_cn, :methodology_cn, :datasets_cn, :results_cn, :critique_cn,
        :domain, :year, :venue, :open_source, :abstract
    )
    ON CONFLICT(paper) DO UPDATE SET
        paper_quality_score  = excluded.paper_quality_score,
        skill_rubric_total   = excluded.skill_rubric_total,
        skill_scores         = excluded.skill_scores,
        skill_reasons        = excluded.skill_reasons,
        comment              = excluded.comment,
        comment_cn           = excluded.comment_cn,
        title_cn             = excluded.title_cn,
        research_question    = excluded.research_question,
        methodology          = excluded.methodology,
        datasets             = excluded.datasets,
        results              = excluded.results,
        critique             = excluded.critique,
        research_question_cn = excluded.research_question_cn,
        methodology_cn       = excluded.methodology_cn,
        datasets_cn          = excluded.datasets_cn,
        results_cn           = excluded.results_cn,
        critique_cn          = excluded.critique_cn,
        domain               = excluded.domain,
        year                 = excluded.year,
        venue                = excluded.venue,
        open_source          = excluded.open_source,
        abstract             = excluded.abstract
    """, {
        "paper":               record.get("paper", ""),
        "thread_id":           record.get("thread_id", ""),
        "timestamp":           record.get("timestamp", ""),
        "prompt_version":      record.get("prompt_version", ""),
        "paper_score":         _to_float(record.get("paper_score")),
        "paper_quality_score": _to_float(record.get("paper_quality_score")),
        "skill_rubric_total":  _to_float(record.get("skill_rubric_total")),
        "skill_scores":        json.dumps(record.get("skill_scores", {}), ensure_ascii=False),
        "skill_reasons":       json.dumps(record.get("skill_reasons", {}), ensure_ascii=False),
        "comment":             record.get("comment", ""),
        "comment_cn":          record.get("comment_cn", ""),
        "title_cn":            record.get("title_cn", ""),
        "research_question":   record.get("research_question", ""),
        "methodology":         record.get("methodology", ""),
        "datasets":            record.get("datasets", ""),
        "results":             record.get("results", ""),
        "critique":            record.get("critique", ""),
        "research_question_cn": record.get("research_question_cn", ""),
        "methodology_cn":      record.get("methodology_cn", ""),
        "datasets_cn":         record.get("datasets_cn", ""),
        "results_cn":          record.get("results_cn", ""),
        "critique_cn":         record.get("critique_cn", ""),
        "domain":              record.get("domain", ""),
        "year":                str(record.get("year", "")),
        "venue":               record.get("venue", ""),
        "open_source":         record.get("open_source", ""),
        "abstract":            record.get("abstract", ""),
    })
    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("skill_scores", "skill_reasons"):
        try:
            d[key] = json.loads(d[key]) if d[key] else {}
        except Exception:
            d[key] = {}
    return d


def get_all_papers() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT rowid, * FROM papers ORDER BY paper_quality_score DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = _row_to_dict(r)
        d["rowid"] = r["rowid"]
        result.append(d)
    return result


def delete_papers_by_rowids(rowids: list) -> int:
    """
    批量删除论文记录（用于清洗非 CS/AI 领域的污染数据）。
    返回实际删除条数。
    """
    if not rowids:
        return 0
    conn = get_connection()
    placeholders = ",".join("?" * len(rowids))
    cur = conn.execute(f"DELETE FROM papers WHERE rowid IN ({placeholders})", rowids)
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


def get_paper_by_rowid(rowid: int) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute("SELECT rowid, * FROM papers WHERE rowid = ?", (rowid,)).fetchone()
    conn.close()
    if not row:
        return None
    d = _row_to_dict(row)
    d["rowid"] = row["rowid"]
    return d


def get_papers_page(sort: str = "hot", page: int = 1, size: int = 10,
                    domain: str = "") -> dict:
    """
    分页获取论文列表，支持三种排序：
      hot     — 热度分（点赞×3 + 评论×2 + 浏览×1 + 质量分×10）
      new     — 按 timestamp 倒序（最新评测）
      quality — 按 paper_quality_score 倒序
    """
    conn   = get_connection()
    offset = (page - 1) * size

    # 热度分：用子查询关联 comments 和 user_behaviors
    hot_score = """
        (p.likes * 3
         + COALESCE((SELECT COUNT(*) FROM comments c WHERE c.paper = p.paper), 0) * 2
         + COALESCE((SELECT COUNT(*) FROM user_behaviors b WHERE b.paper = p.paper AND b.action='view'), 0)
         + p.paper_quality_score * 10
        )
    """

    order_map = {
        "hot":     f"{hot_score} DESC",
        "new":     "p.timestamp DESC",
        "quality": "p.paper_quality_score DESC",
    }
    order_clause = order_map.get(sort, order_map["hot"])

    where  = "WHERE p.domain = ?" if domain else ""
    params = [domain] if domain else []

    total_row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM papers p {where}", params
    ).fetchone()
    total = total_row["cnt"] if total_row else 0

    rows = conn.execute(
        f"""SELECT p.*, {hot_score} as hot_score
            FROM papers p
            {where}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?""",
        params + [size, offset]
    ).fetchall()
    conn.close()

    papers = []
    for r in rows:
        d = _row_to_dict(r)
        d["hot_score"] = round(r["hot_score"], 2)
        papers.append(d)

    return {
        "total":    total,
        "page":     page,
        "size":     size,
        "has_more": (page * size) < total,
        "papers":   papers,
    }


def get_paper(name: str) -> Optional[dict]:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM papers WHERE LOWER(paper) LIKE ?",
        (f"%{name.lower()}%",)
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None
