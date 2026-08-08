"""CommSpec RAG — 统一服务入口."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time as _time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.auth import APIKeyMiddleware
from src.api.rest.admin_router import admin_router, auth_router
from src.api.rest.feedback import router as feedback_router
from src.api.rest.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    register_exception_handlers,
)
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


def _start_release_monitor() -> threading.Thread | None:
    """启动后台 Release 变更监控线程 (Task 12).

    每 N 分钟扫描文档目录, 检测新增/修改, 自动记录变更日志。
    不自动触发摄入 (auto_index=False), 仅检测+告警。
    """
    interval_minutes = getattr(settings, 'release_monitor_interval_minutes', 0) or 120
    if interval_minutes <= 0:
        logger.info("Release 监控已禁用 (interval=0)")
        return None

    from src.ingestion.release_monitor import ReleaseMonitor

    monitor = ReleaseMonitor()
    stop_event = threading.Event()

    def _run():
        # 启动后首次延迟 30s 再检测, 让服务先完成初始化
        _time.sleep(30)
        logger.info("Release 监控已启动 (间隔=%dmin)", interval_minutes)
        while not stop_event.is_set():
            try:
                report = monitor.check_and_process(auto_index=False)
                if report.has_changes:
                    logger.warning(
                        "📢 Release 变更检测: +%d 新增, ~%d 修改, -%d 删除",
                        len(report.new_files), len(report.modified_files),
                        len(report.deleted_keys),
                    )
                    for f in report.new_files[:5]:
                        logger.info("  新增: %s (%s, %s)", f.path.name, f.spec_number, f.release)
            except Exception as e:
                logger.warning("Release 监控检测失败: %s", e)
            stop_event.wait(interval_minutes * 60)

    t = threading.Thread(target=_run, daemon=True, name="release-monitor")
    t.start()
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _pipeline
    logger.info("=" * 50)
    logger.info("CommSpec RAG 启动 — LLM: %s, 向量库: %s", settings.llm_model, settings.vector_db)

    # ── Task 6: 在线搜索配置验证 ──
    _validate_online_search_config()
    # ── Task 8: 嵌入缓存状态 ──
    _log_embedding_cache_status()
    # ── Task 13: API Key 认证状态 ──
    if not os.getenv("API_KEYS", "").strip():
        logger.warning(
            "⚠️ 未配置 API_KEYS — 所有 API 端点将免认证放行。"
            "生产环境必须设置 API_KEYS (逗号分隔), 否则任何人可调用 /ask、/search、/documents 等端点。"
        )

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

    # ── Task 12: 启动 Release 监控 ──
    _monitor_thread = _start_release_monitor()

    yield
    if _store:
        _store.disconnect()
    logger.info("服务已关闭")


app = FastAPI(
    title="CommSpec RAG API",
    description="""通信规范（3GPP / O-RAN）检索增强生成系统 — 支持语义搜索、RAG 问答、文档管理。

## 主要功能
- **Search**: Dense 向量检索 + BM25 混合检索
- **Ask**: RAG 增强的 LLM 问答, 含幻觉验证
- **Documents**: 规范文档 CRUD 管理
- **Ingestion**: DOCX 摄入管线 (下载→转换→分块→嵌入)
- **Streaming**: SSE 流式生成, 逐 token 推送
""",
    version=__version__,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Search", "description": "语义检索 + 混合检索"},
        {"name": "Ask", "description": "RAG 问答 + 流式生成"},
        {"name": "Documents", "description": "规范文档 CRUD"},
        {"name": "System", "description": "健康检查 + 系统统计"},
    ],
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)

register_exception_handlers(app)

app.include_router(router)
app.include_router(admin_router)
app.include_router(auth_router)
app.include_router(feedback_router)

# Prometheus /metrics 端点
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)


# ── 启动辅助函数 ──

def _validate_online_search_config() -> None:
    """Task 6: 启动时验证在线搜索配置并记录状态."""
    if not settings.enable_online_search:
        logger.info("在线搜索: 已禁用 (ENABLE_ONLINE_SEARCH=false)")
        return

    google_ok = bool(settings.google_api_key and settings.google_cse_id)
    tspec_ok = bool(settings.tspec_llm_url)

    if not google_ok and not tspec_ok:
        logger.warning(
            "在线搜索已启用但未配置任何数据源! "
            "设置 GOOGLE_API_KEY + GOOGLE_CSE_ID 或 TSPEC_LLM_URL"
        )
        return

    sources = []
    if google_ok:
        sources.append("Google CSE")
    if tspec_ok:
        sources.append(f"TSpec-LLM ({settings.tspec_llm_url})")
    logger.info(
        "在线搜索: 已启用 | 数据源: %s | 触发阈值: score<%.2f 或 count<%d",
        ", ".join(sources),
        settings.online_score_threshold,
        5,  # OnlineSupplement 默认 count_threshold
    )


def _log_embedding_cache_status() -> None:
    """Task 8: 启动时输出嵌入缓存统计."""
    try:
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache()
        st = cache.stats()
        if st["total_entries"] > 0:
            logger.info(
                "嵌入缓存: %d 条 (%s MB) — %s",
                st["total_entries"], st["size_mb"], st["db_path"],
            )
        else:
            logger.info("嵌入缓存: 空 (首次运行将自动填充)")
    except Exception as e:
        logger.debug("嵌入缓存状态读取失败: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )
