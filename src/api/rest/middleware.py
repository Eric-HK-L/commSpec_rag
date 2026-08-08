"""FastAPI 全局异常处理 + 请求日志 + 速率限制中间件."""

from __future__ import annotations

import logging
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.rest.schemas import ErrorDetail
from src.config import settings

logger = logging.getLogger(__name__)


# ── 异常处理器 (注册到 FastAPI app) ──

def register_exception_handlers(app):
    """注册全局异常处理器到 FastAPI app 实例."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning("参数错误: %s", exc)
        return JSONResponse(
            status_code=400,
            content=ErrorDetail(
                error_code="INVALID_PARAMETER",
                detail=str(exc),
                suggestion="请检查请求参数格式和范围",
            ).model_dump(),
        )

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning("资源不存在: %s", exc)
        return JSONResponse(
            status_code=404,
            content=ErrorDetail(
                error_code="NOT_FOUND",
                detail=str(exc),
                suggestion="请确认资源 ID 或路径是否正确",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(
            "未处理异常 %s %s: %s",
            request.method, request.url.path, exc, exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorDetail(
                error_code="INTERNAL_ERROR",
                detail="服务内部错误，请稍后重试",
                suggestion="请联系管理员或查看服务日志",
            ).model_dump(),
        )


# ── 请求日志中间件 ──

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的方法、路径、耗时."""

    async def dispatch(self, request: Request, call_next):
        import time
        t0 = time.time()
        response = await call_next(request)
        dt = (time.time() - t0) * 1000
        logger.info(
            "%s %s → %d (%.0fms)",
            request.method, request.url.path, response.status_code, dt,
        )
        return response


# ── 速率限制中间件 ──


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按 IP 的滑动窗口限流 — 仅作用于 LLM 相关端点 (/ask, /search)."""

    def __init__(self, app, rpm: int | None = None, enabled: bool | None = None):
        super().__init__(app)
        self._enabled = settings.rate_limit_enabled if enabled is None else enabled
        self._rpm = settings.rate_limit_rpm if rpm is None else rpm
        self._window = 60.0
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            return await call_next(request)
        path = request.url.path
        if not (path.startswith("/api/v1/ask") or path.startswith("/api/v1/search")):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            times = [t for t in self._hits.get(client, []) if now - t < self._window]
            if len(times) >= self._rpm:
                return JSONResponse(
                    status_code=429,
                    content={"error": "请求过于频繁，请稍后重试"},
                )
            times.append(now)
            self._hits[client] = times
        return await call_next(request)
