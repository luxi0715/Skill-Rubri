"""
test_auth.py
-------------
测试用户注册、登录、JWT鉴权（SQLite层 + auth模块）
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import db as db_module
from auth import hash_password, verify_password, create_token, decode_token


# ── Fixture：每个测试用独立的临时数据库 ─────────────────────
@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_auth.db")
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    yield


# ── 密码哈希 ────────────────────────────────────────────────
def test_hash_and_verify_password():
    hashed = hash_password("mypassword123")
    assert verify_password("mypassword123", hashed)
    assert not verify_password("wrongpassword", hashed)


def test_different_passwords_produce_different_hashes():
    h1 = hash_password("password1")
    h2 = hash_password("password1")
    assert h1 != h2  # bcrypt 每次 salt 不同


# ── JWT Token ───────────────────────────────────────────────
def test_create_and_decode_token():
    token   = create_token(1, "test@example.com", "张三")
    payload = decode_token(token)
    assert payload["sub"]      == "1"
    assert payload["email"]    == "test@example.com"
    assert payload["nickname"] == "张三"


def test_invalid_token_raises():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        decode_token("invalid.token.here")
    assert exc_info.value.status_code == 401


# ── 用户注册 ────────────────────────────────────────────────
def test_register_user():
    user = db_module.create_user("alice@example.com", "Alice", hash_password("pass123"))
    assert user["id"] is not None
    assert user["email"] == "alice@example.com"
    assert user["nickname"] == "Alice"


def test_register_duplicate_email_raises():
    db_module.create_user("bob@example.com", "Bob", hash_password("pass123"))
    with pytest.raises(Exception):
        db_module.create_user("bob@example.com", "Bob2", hash_password("pass456"))


# ── 用户登录（查询）───────────────────────────────────────────
def test_get_user_by_email():
    hashed = hash_password("secret")
    db_module.create_user("carol@example.com", "Carol", hashed)
    user = db_module.get_user_by_email("carol@example.com")
    assert user is not None
    assert user["nickname"] == "Carol"
    assert verify_password("secret", user["password_hash"])


def test_get_nonexistent_user_returns_none():
    assert db_module.get_user_by_email("nobody@example.com") is None


def test_email_case_insensitive():
    db_module.create_user("Dave@Example.COM", "Dave", hash_password("pass"))
    user = db_module.get_user_by_email("dave@example.com")
    assert user is not None


# ── 行为记录 ────────────────────────────────────────────────
def test_record_and_count_views():
    db_module.record_behavior("test_paper.pdf", "view", user_id=None, anon_id="anon-123")
    db_module.record_behavior("test_paper.pdf", "view", user_id=None, anon_id="anon-456")
    db_module.record_behavior("test_paper.pdf", "like", user_id=None, anon_id="anon-123")
    count = db_module.get_paper_view_count("test_paper.pdf")
    assert count == 2  # 只统计 view，不含 like


def test_get_user_by_id():
    user = db_module.create_user("eve@example.com", "Eve", hash_password("pass"))
    found = db_module.get_user_by_id(user["id"])
    assert found["nickname"] == "Eve"
    assert "password_hash" not in found  # 不暴露密码哈希
