from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("kb_web_agent.auth")

Role = Literal["admin", "user"]

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class User:
    username: str
    role: Role
    departments: list[str] = field(default_factory=list)
    # [] 表示 admin 可访问全部 department


# ---------------------------------------------------------------------------
# 简易内存用户表（生产中应替换为数据库）
# ---------------------------------------------------------------------------

_USERS: dict[str, dict] = {
    "admin": {
        "password": os.environ.get("ADMIN_PASSWORD", "admin123"),
        "role": "admin",
        "departments": [],
    },
    "user": {
        "password": os.environ.get("USER_PASSWORD", "user123"),
        "role": "user",
        "departments": ["default"],
    },
}


def verify_password(username: str, password: str) -> User | None:
    record = _USERS.get(username)
    if record and record["password"] == password:
        return User(
            username=username,
            role=record["role"],
            departments=record["departments"],
        )
    return None


# ---------------------------------------------------------------------------
# JWT 签发 / 验证
# ---------------------------------------------------------------------------

_JWT_ALGORITHM = "HS256"
_jwt_secret: str | None = None


def _get_secret() -> str:
    global _jwt_secret
    if _jwt_secret is None:
        _jwt_secret = os.environ.get("JWT_SECRET") or secrets.token_hex(32)
        if not os.environ.get("JWT_SECRET"):
            logger.warning("[Auth] JWT_SECRET 未设置，使用随机密钥（重启后 token 失效）")
    return _jwt_secret


def create_token(user: User, expire_minutes: int = 480) -> str:
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose 未安装，请执行 pip install python-jose[cryptography]")

    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {
        "sub": user.username,
        "role": user.role,
        "departments": user.departments,
        "exp": expire,
    }
    return jwt.encode(payload, _get_secret(), algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> User:
    try:
        from jose import JWTError, jwt
    except ImportError:
        raise RuntimeError("python-jose 未安装")

    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[_JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效或已过期: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(
        username=payload["sub"],
        role=payload["role"],
        departments=payload.get("departments", []),
    )


# ---------------------------------------------------------------------------
# FastAPI 依赖
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请提供 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def get_department_filter(user: User = Depends(get_current_user)) -> list[str] | None:
    """根据用户角色返回 department_filter：admin 返回 None（查全库），user 返回其 departments。"""
    if user.role == "admin":
        return None
    return user.departments or ["default"]
