"""
migrate_to_postgres.py
-----------------------
把 SQLite（papers.db）数据全量迁移到 PostgreSQL。

使用前提：
  1. 已安装 PostgreSQL，并创建数据库：
     CREATE DATABASE paperdb;
  2. 配置 .env：
     PG_DSN=postgresql://user:password@localhost:5432/paperdb

运行：
  python migrate_to_postgres.py
"""

import json
import os
import sqlite3
import sys

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "papers.db")
PG_DSN   = os.getenv("PG_DSN", "")

# ── DDL（与 db.py 保持一致）────────────────────────────────
CREATE_PAPERS = """
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
);
"""

CREATE_COMMENTS = """
CREATE TABLE IF NOT EXISTS comments (
    id         SERIAL PRIMARY KEY,
    paper      TEXT NOT NULL,
    nickname   TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    likes      INTEGER DEFAULT 0,
    user_id    INTEGER DEFAULT NULL
);
"""

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    nickname      TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    is_active     INTEGER DEFAULT 1
);
"""

CREATE_BEHAVIORS = """
CREATE TABLE IF NOT EXISTS user_behaviors (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER,
    anon_id    TEXT,
    paper      TEXT NOT NULL,
    action     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

CREATE_IMAGES = """
CREATE TABLE IF NOT EXISTS paper_images (
    id         SERIAL PRIMARY KEY,
    paper      TEXT NOT NULL,
    image_path TEXT NOT NULL,
    caption    TEXT,
    order_idx  INTEGER DEFAULT 0
);
"""


def get_pg():
    import psycopg2
    return psycopg2.connect(PG_DSN)


def init_pg(conn):
    cur = conn.cursor()
    for ddl in [CREATE_PAPERS, CREATE_COMMENTS, CREATE_USERS,
                CREATE_BEHAVIORS, CREATE_IMAGES]:
        cur.execute(ddl)
    conn.commit()
    cur.close()
    print("  PostgreSQL 表结构初始化完成")


def migrate_table(sqlite_conn, pg_conn, table: str, pk: str):
    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(f"SELECT * FROM {table}")
    rows = sqlite_cur.fetchall()
    cols = [d[0] for d in sqlite_cur.description]

    if not rows:
        print(f"  {table}: 0 行，跳过")
        return

    pg_cur = pg_conn.cursor()
    placeholders = ",".join(["%s"] * len(cols))
    col_names    = ",".join(cols)

    upsert_sql = (
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk}) DO NOTHING"
    )

    pg_cur.executemany(upsert_sql, rows)
    pg_conn.commit()
    pg_cur.close()
    print(f"  {table}: {len(rows)} 行迁移完成")


def main():
    if not PG_DSN:
        print("[ERROR] 未设置 PG_DSN，请在 .env 中配置：")
        print("  PG_DSN=postgresql://user:password@localhost:5432/paperdb")
        sys.exit(1)

    print("=" * 50)
    print("SQLite → PostgreSQL 全量迁移")
    print(f"  源：{DB_PATH}")
    print(f"  目标：{PG_DSN[:30]}...")
    print("=" * 50)

    sqlite_conn = sqlite3.connect(DB_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    try:
        pg_conn = get_pg()
    except Exception as e:
        print(f"[ERROR] PostgreSQL 连接失败：{e}")
        sys.exit(1)

    init_pg(pg_conn)

    # papers 表：主键 paper（TEXT）
    migrate_table(sqlite_conn, pg_conn, "papers",        "paper")
    # comments/users/behaviors 主键是自增 id，用 DO NOTHING 防重复
    migrate_table(sqlite_conn, pg_conn, "comments",      "id")
    migrate_table(sqlite_conn, pg_conn, "users",         "id")
    migrate_table(sqlite_conn, pg_conn, "user_behaviors","id")
    migrate_table(sqlite_conn, pg_conn, "paper_images",  "id")

    sqlite_conn.close()
    pg_conn.close()
    print("\n[done] 迁移完成。")


if __name__ == "__main__":
    main()
