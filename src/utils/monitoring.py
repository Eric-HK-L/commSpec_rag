"""Prometheus 监控 — 指标定义 + /metrics 端点 + 轻量 HTTP 中间件."""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  指标定义
# ═══════════════════════════════════════════

# HTTP 请求
http_requests_total = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# RAG Pipeline
rag_search_total = Counter(
    "rag_search_total",
    "Total RAG search calls",
    ["status"],  # success / error
)

rag_search_duration_seconds = Histogram(
    "rag_search_duration_seconds",
    "Search latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

rag_ask_total = Counter(
    "rag_ask_total",
    "Total RAG ask (full pipeline) calls",
    ["status"],
)

rag_ask_duration_seconds = Histogram(
    "rag_ask_duration_seconds",
    "Full pipeline latency (search + LLM) in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0),
)

# LLM
rag_llm_tokens_total = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["type"],  # prompt / completion
)

rag_llm_call_duration_seconds = Histogram(
    "rag_llm_call_duration_seconds",
    "LLM API call duration in seconds",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
)

# Retrieval quality
rag_retrieval_results_count = Histogram(
    "rag_retrieval_results_count",
    "Number of results per retrieval",
    buckets=(1, 3, 5, 10, 20, 50),
)

rag_retrieval_avg_score = Histogram(
    "rag_retrieval_avg_score",
    "Distribution of average retrieval score of top-k results",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")),
)

# 交叉引用 & 多跳
rag_multi_hop_triggered_total = Counter(
    "rag_multi_hop_triggered_total",
    "Times multi-hop retrieval was triggered",
)

rag_cross_refs_resolved_total = Counter(
    "rag_cross_refs_resolved_total",
    "Total cross-references resolved",
)

# 错误
rag_errors_total = Counter(
    "rag_errors_total",
    "Total errors by type",
    ["error_type"],
)


# ═══════════════════════════════════════════
#  HTTP 监控中间件
# ═══════════════════════════════════════════

class PrometheusMiddleware(BaseHTTPMiddleware):
    """自动记录每个 HTTP 请求的计数与延迟."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过 /metrics 自身避免自循环
        if request.url.path == "/metrics":
            return await call_next(request)

        t0 = time.time()
        response = await call_next(request)
        dt = time.time() - t0

        endpoint = _route_label(request)
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(dt)

        return response


def _route_label(request: Request) -> str:
    """将路径转为稳定的标签值，避免高基数.

    例如 /api/v1/ask → ask, /api/v1/documents/xxx → documents/:id
    """
    path = request.url.path
    # 匹配 FastAPI 注册的路由路径
    if hasattr(request.scope, "get") and callable(request.scope.get):
        route = request.scope.get("route")
        if route and hasattr(route, "path"):
            return route.path.replace("/api/v1/", "").replace("/", "_")
    # Fallback: 简单归一化
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "v1":
        label = parts[2]  # 取 /api/v1/<resource>
        return label
    return path.strip("/").replace("/", "_") or "root"


# ═══════════════════════════════════════════
#  /metrics 端点
# ═══════════════════════════════════════════

async def metrics_endpoint() -> Response:
    """Prometheus /metrics 端点."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ═══════════════════════════════════════════
#  便捷埋点工具
# ═══════════════════════════════════════════

def record_search(results_count: int, avg_score: float, duration_s: float, success: bool = True):
    """记录一次检索操作."""
    status = "success" if success else "error"
    rag_search_total.labels(status=status).inc()
    rag_search_duration_seconds.observe(duration_s)
    rag_retrieval_results_count.observe(results_count)
    rag_retrieval_avg_score.observe(avg_score)


def record_ask(duration_s: float, success: bool = True):
    """记录一次完整 RAG 问答."""
    rag_ask_total.labels(status="success" if success else "error").inc()
    rag_ask_duration_seconds.observe(duration_s)


def record_llm_call(prompt_tokens: int, completion_tokens: int, duration_s: float):
    """记录一次 LLM 调用."""
    rag_llm_tokens_total.labels(type="prompt").inc(prompt_tokens)
    rag_llm_tokens_total.labels(type="completion").inc(completion_tokens)
    rag_llm_call_duration_seconds.observe(duration_s)


def record_error(error_type: str):
    """记录错误."""
    rag_errors_total.labels(error_type=error_type).inc()


def record_multi_hop():
    """记录多跳触发."""
    rag_multi_hop_triggered_total.inc()


def record_cross_ref(count: int = 1):
    """记录交叉引用解析."""
    rag_cross_refs_resolved_total.inc(count)
