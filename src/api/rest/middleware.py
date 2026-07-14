"""FastAPI 全局异常处理中间件."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.rest.schemas import ErrorDetail

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
        logger.error("未处理异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorDetail(
                error_code="INTERNAL_ERROR",
                detail=str(exc),
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
