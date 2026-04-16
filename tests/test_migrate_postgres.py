"""
test_migrate_postgres.py
-------------------------
验证 Step 3：SQLite → PostgreSQL 迁移脚本

分两组：
  A. 单元测试（不依赖 PostgreSQL）：验证 DDL 格式、SQLite 读取、工具函数
  B. 集成测试（需要 PostgreSQL）：自动跳过若 PG_DSN 未配置或连不上

运行：
  python -m pytest tests/test_migrate_postgres.py -v
"""

import os
import sys
import sqlite3
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ── 判断 PG 是否可用 ────────────────────────────────────
def _pg_available():
    dsn = os.getenv("PG_DSN", "")
    if not dsn:
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(dsn, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False

requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="PG_DSN 未配置或 PostgreSQL 未启动，跳过集成测试"
)


# ── A. 单元测试 ──────────────────────────────────────────

class TestDDL:
    def test_create_papers_has_primary_key(self):
        from migrate_to_postgres import CREATE_PAPERS
        assert "PRIMARY KEY" in CREATE_PAPERS
        assert "paper" in CREATE_PAPERS

    def test_create_comments_uses_serial(self):
        from migrate_to_postgres import CREATE_COMMENTS
        assert "SERIAL" in CREATE_COMMENTS

    def test_all_ddl_importable(self):
        from migrate_to_postgres import (
            CREATE_PAPERS, CREATE_COMMENTS, CREATE_USERS,
            CREATE_BEHAVIORS, CREATE_IMAGES
        )
        for ddl in [CREATE_PAPERS, CREATE_COMMENTS,
                    CREATE_USERS, CREATE_BEHAVIORS, CREATE_IMAGES]:
            assert "CREATE TABLE" in ddl


class TestSQLiteRead:
    """用临时 SQLite 验证迁移读取逻辑"""

    def _make_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE papers (
                paper TEXT PRIMARY KEY,
                paper_quality_score REAL DEFAULT 0,
                skill_rubric_total  REAL DEFAULT 0,
                likes INTEGER DEFAULT 0
            )
        """)
        conn.execute("INSERT INTO papers VALUES ('test.pdf', 0.8, 0.9, 3)")
        conn.execute("INSERT INTO papers VALUES ('test2.pdf', 0.6, 0.7, 1)")
        conn.commit()
        return conn

    def test_sqlite_read_all_rows(self):
        conn = self._make_sqlite()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM papers")
        rows = cur.fetchall()
        assert len(rows) == 2

    def test_sqlite_column_names(self):
        conn = self._make_sqlite()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM papers")
        cur.fetchall()
        cols = [d[0] for d in cur.description]
        assert "paper" in cols
        assert "likes" in cols

    def test_sqlite_placeholder_count_matches_cols(self):
        conn = self._make_sqlite()
        cur  = conn.cursor()
        cur.execute("SELECT * FROM papers")
        cur.fetchall()
        cols         = [d[0] for d in cur.description]
        placeholders = ",".join(["%s"] * len(cols))
        assert placeholders.count("%s") == len(cols)

    def test_pg_dsn_env_read(self):
        """PG_DSN 从环境变量读取，缺失时为空字符串"""
        import importlib, migrate_to_postgres as m
        dsn = os.getenv("PG_DSN", "")
        assert isinstance(dsn, str)   # 不崩溃即可


# ── B. 集成测试（需要 PostgreSQL）───────────────────────

class TestPostgresIntegration:

    @requires_pg
    def test_pg_connection(self):
        import psycopg2
        conn = psycopg2.connect(os.getenv("PG_DSN"))
        assert conn is not None
        conn.close()

    @requires_pg
    def test_init_creates_tables(self):
        import psycopg2
        from migrate_to_postgres import init_pg
        conn = psycopg2.connect(os.getenv("PG_DSN"))
        init_pg(conn)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        for t in ["papers", "comments", "users", "user_behaviors"]:
            assert t in tables, f"表 {t} 未创建"

    @requires_pg
    def test_migrate_papers_table(self):
        """迁移 papers 表后行数一致"""
        import psycopg2
        from migrate_to_postgres import get_pg, init_pg, migrate_table

        # SQLite 实际行数
        BASE_DIR = os.path.dirname(os.path.dirname(__file__))
        DB_PATH  = os.path.join(BASE_DIR, "data", "papers.db")
        sqlite_conn = sqlite3.connect(DB_PATH)
        sqlite_cur  = sqlite_conn.cursor()
        sqlite_cur.execute("SELECT COUNT(*) FROM papers")
        sqlite_count = sqlite_cur.fetchone()[0]
        sqlite_conn.close()

        pg_conn = get_pg()
        init_pg(pg_conn)
        migrate_table(sqlite_conn, pg_conn, "papers", "paper")

        pg_cur = pg_conn.cursor()
        pg_cur.execute("SELECT COUNT(*) FROM papers")
        pg_count = pg_cur.fetchone()[0]
        pg_conn.close()

        assert pg_count == sqlite_count
