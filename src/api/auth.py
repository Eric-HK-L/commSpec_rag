"""API Key 认证中间件 — 校验 X-API-Key header."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

# 从环境变量加载有效 Key 列表（逗号分隔）
_VALID_KEYS: set[str] = set(
    k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()
)

# 无需认证的路径前缀
_SKIP_PATHS = {"/api/v1/health", "/docs", "/redoc", "/openapi.json", "/api/v1/mcp"}


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
