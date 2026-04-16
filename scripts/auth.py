"""
auth.py
--------
用户认证模块：密码哈希 + JWT Token

依赖：
    pip install bcrypt python-jose[cryptography]
"""

import os
from datetime import datetime, timedelta

import bcrypt
from jose import JWTError, jwt
from fastapi import HTTPException, Header
from dotenv import load_dotenv
from typing import Optional

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

SECRET_KEY = os.getenv("JWT_SECRET", "changeme-use-a-real-secret-in-production")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_DAYS = 30


# ── 密码 ───────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── JWT ────────────────────────────────────────────────────
def create_token(user_id: int, email: str, nickname: str) -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {
        "sub":      str(user_id),
        "email":    email,
        "nickname": nickname,
        "exp":      expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """解码 Token，失败抛 HTTPException 401"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def get_current_user(authorization: str = Header(None)) -> dict:
    """FastAPI 依赖注入：从 Authorization: Bearer <token> 解析用户"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization[7:]
    return decode_token(token)


def get_optional_user(authorization: str = Header(None)) -> Optional[dict]:
    """可选登录：有 Token 就解析，没有返回 None（不报错）"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return decode_token(authorization[7:])
    except HTTPException:
        return None
