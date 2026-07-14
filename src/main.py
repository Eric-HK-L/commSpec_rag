"""3GPP RAG 融合项目 — 统一服务入口."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.auth import APIKeyMiddleware
from src.api.rest.admin_router import admin_router
from src.api.rest.feedback import router as feedback_router
from src.api.rest.middleware import RequestLoggingMiddleware, register_exception_handlers
from src.api.rest.router import router, set_pipeline
from src.config import settings
from src.generator.llm_client import LLMClient
from src.generator.pipeline import RAGPipeline
from src.retriever.vector_store import VectorStore
from src.utils.monitoring import PrometheusMiddleware, metrics_endpoint

# ── 日志 ──
log_dir = settings.log_abs_file.parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.log_abs_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

_store: VectorStore | None = None
_pipeline: RAGPipeline | None = None


def init_vector_store() -> VectorStore:
    """初始化 Milvus 向量数据库连接."""
    from src.retriever.milvus_store import MilvusStore
    store = MilvusStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection_name,
    )
    store.connect()
    logger.info("Milvus 连接成功 (Dense + BM25 混合检索)")

    # 自动加载 BM25 索引
    if store.load_bm25():
        logger.info("BM25 索引已加载 (%d 条)", store.bm25_count)
    else:
        logger.warning("BM25 索引未找到, 仅 Dense 检索可用")

    return store


def init_pipeline(store: VectorStore) -> RAGPipeline:
    """初始化 RAG 流水线."""
    llm = LLMClient()
    pipeline = RAGPipeline(vector_store=store, llm_client=llm)
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _pipeline
    logger.info("=" * 50)
    logger.info("3GPP RAG 启动 — LLM: %s, 向量库: %s", settings.llm_model, settings.vector_db)
    _store = init_vector_store()
    try:
        _store.create_collection(drop_existing=False)
    except Exception as e:
        logger.warning("集合初始化: %s", e)
    _pipeline = init_pipeline(_store)
    set_pipeline(_pipeline)
    logger.info("服务就绪 — %d 条规范记录", _store.count)
    # 预热嵌入模型 (避免首次查询时加载延迟)
    _pipeline._warmup()
    yield
    if _store:
        _store.disconnect()
    logger.info("服务已关闭")


app = FastAPI(
    title="3GPP RAG API",
    description="""3GPP 规范检索增强生成系统 — 支持语义搜索、RAG 问答、文档管理。

## 主要功能
- **Search**: Dense 向量检索 + BM25 混合检索
- **Ask**: RAG 增强的 LLM 问答, 含幻觉验证
- **Documents**: 3GPP 规范文档 CRUD 管理
- **Ingestion**: DOCX 摄入管线 (下载→转换→分块→嵌入)
- **Streaming**: SSE 流式生成, 逐 token 推送
""",
    version="0.2.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Search", "description": "语义检索 + 混合检索"},
        {"name": "Ask", "description": "RAG 问答 + 流式生成"},
        {"name": "Documents", "description": "规范文档 CRUD"},
        {"name": "System", "description": "健康检查 + 系统统计"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(APIKeyMiddleware)

register_exception_handlers(app)

app.include_router(router)
app.include_router(admin_router)
app.include_router(feedback_router)

# Prometheus /metrics 端点
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )
