"""认证模块 — API Key 中间件 + 管理后台会话 (HMAC 签名 Cookie)."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

# 从环境变量加载有效 Key 列表（逗号分隔）
_VALID_KEYS: set[str] = set(
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
)

# 无需认证的路径前缀
_SKIP_PATHS = {"/api/v1/health", "/docs", "/redoc", "/openapi.json"}

# 管理后台会话
ADMIN_SESSION_COOKIE = "admin_session"
_random_secret: str | None = None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """验证 X-API-Key header，不通过返回 401."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 跳过公开路径
        for prefix in _SKIP_PATHS:
            if path.startswith(prefix):
                return await call_next(request)

        # 未配置任何 Key → 跳过认证
        if not _VALID_KEYS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if api_key not in _VALID_KEYS:
            raise HTTPException(status_code=401, detail="无效或缺失 API Key")

        return await call_next(request)


# ── 管理后台会话 (HMAC 签名 Cookie) ──


def _session_secret() -> bytes:
    """返回会话签名密钥 — 显式配置优先, 否则进程内随机生成."""
    global _random_secret
    configured = settings.admin_session_secret
    if configured:
        return configured.encode()
    if _random_secret is None:
        _random_secret = secrets.token_hex(32)
    return _random_secret.encode()


def _sign_session(username: str, expires_ts: int) -> str:
    payload = f"{username}:{expires_ts}"
    sig = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def _verify_session(token: str) -> str | None:
    """校验会话 Cookie, 返回用户名; 无效/过期返回 None."""
    if not token:
        return None
    try:
        payload, sig = token.rsplit(":", 1)
        expected = hmac.new(_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        username, exp = payload.split(":", 1)
        if int(exp) < time.time():
            return None
        return username
    except (ValueError, TypeError):
        return None


def create_admin_session(response: Response, username: str) -> None:
    """在响应上写入 httpOnly 会话 Cookie.

    生产环境 (HTTPS) 应设置 ADMIN_COOKIE_SECURE=true; Secure 标记会
    使本地 http://localhost 开发环境无法登录, 故默认关闭。
    """
    ttl = settings.admin_session_ttl_hours * 3600
    response.set_cookie(
        ADMIN_SESSION_COOKIE,
        _sign_session(username, int(time.time()) + ttl),
        max_age=ttl,
        httponly=True,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        path="/",
    )


def clear_admin_session(response: Response) -> None:
    """清除会话 Cookie."""
    response.delete_cookie(ADMIN_SESSION_COOKIE, path="/")


def require_admin(request: Request) -> None:
    """FastAPI 依赖 — 校验管理后台会话."""
    token = request.cookies.get(ADMIN_SESSION_COOKIE, "")
    if _verify_session(token) is None:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
