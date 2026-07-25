"""Phase 4 管理 API — 知识库运维端点."""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.rest.router import get_pipeline
from src.api.rest.schemas import APIResponse
from src.config import settings

logger = logging.getLogger(__name__)

admin_router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])

# 项目根目录 (跨平台)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 摄入日志路径 — 跨平台: macOS/Linux /tmp, Windows %TEMP%
INGEST_LOG_PATH = Path(tempfile.gettempdir()) / "bulk_ingest_v3.log"
# Manifest 路径 — 从 settings 读取
MANIFEST_PATH = settings.manifest_path
# 后台摄入进程 PID 跟踪
_ingest_process: subprocess.Popen | None = None
# 服务启动时间（用于 uptime 计算）
_start_time = time.time()


# ── 响应模型 ──

class AdminStats(BaseModel):
    """增强系统统计（含 manifest + BM25 状态）."""
    total_docs: int
    total_chunks: int
    releases: dict[str, int]
    series_chunk_distribution: dict[str, int]
    vector_db: str
    bm25_loaded: bool = False
    bm25_count: int = 0
    manifest_records: int = 0
    last_ingestion: str | None = None


class IngestTriggerResponse(BaseModel):
    """摄入触发结果."""
    accepted: bool
    message: str
    mode: str = "incremental"
    pid: int | None = None


class IngestStatus(BaseModel):
    """摄入运行状态."""
    running: bool
    pid: int | None = None
    log_tail: list[str] = Field(default_factory=list)
    last_ingestion_at: str | None = None


class ManifestItem(BaseModel):
    """单条 manifest 记录."""
    key: str
    spec_number: str
    release: str
    latest_version: str
    file_path: str
    sha256: str
    chunk_count: int
    ingested_at: str


# ── 端点实现 ──

@admin_router.get("/stats", response_model=APIResponse[AdminStats])
async def admin_stats() -> APIResponse[AdminStats]:
    """增强统计：含 manifest 信息、BM25 状态、最近摄入记录."""
    pipeline = get_pipeline()
    store = pipeline._store

    # 基础统计
    total_chunks = store.count
    doc_map = _build_doc_map(pipeline)

    releases: dict[str, int] = {}
    series_dist: dict[str, int] = {}
    for doc in doc_map.values():
        if doc.get("release"):
            releases[doc["release"]] = releases.get(doc["release"], 0) + 1
        series = str(doc.get("series", 0))
        series_dist[series] = series_dist.get(series, 0) + doc.get("chunk_count", 0)

    # BM25 状态
    bm25_loaded = getattr(store, "_bm25", None) is not None and store._bm25.is_loaded
    bm25_count = store.bm25_count if bm25_loaded else 0

    # Manifest 状态
    manifest_records = 0
    last_ingestion = None
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text("utf-8"))
            specs = data.get("specs", {})
            manifest_records = len(specs)
            # 找最新摄入时间
            times = [v.get("ingested_at", "") for v in specs.values()]
            times = [t for t in times if t]
            if times:
                last_ingestion = max(times)
        except Exception:
            pass

    return APIResponse.ok(AdminStats(
        total_docs=len(doc_map),
        total_chunks=total_chunks,
        releases=releases,
        series_chunk_distribution=series_dist,
        vector_db=store.__class__.__name__,
        bm25_loaded=bm25_loaded,
        bm25_count=bm25_count,
        manifest_records=manifest_records,
        last_ingestion=last_ingestion,
    ))


@admin_router.post("/ingest/trigger", response_model=APIResponse[IngestTriggerResponse])
async def trigger_ingestion(
    mode: str = "incremental",
) -> APIResponse[IngestTriggerResponse]:
    """触发摄入任务（后台子进程）.

    - mode=incremental: 增量模式（默认）
    - mode=full: 全量重建
    """
    global _ingest_process

    # 检查是否已有任务在运行
    if _ingest_process is not None and _ingest_process.poll() is None:
        return APIResponse.ok(IngestTriggerResponse(
            accepted=False,
            message=f"摄入任务已在运行中 (PID: {_ingest_process.pid})",
            mode=mode,
            pid=_ingest_process.pid,
        ))

    # 构建命令
    script = _PROJECT_ROOT / "scripts" / "bulk_ingest.py"
    venv_python = sys.executable  # 使用当前 venv 的 Python (跨平台)

    cmd = [venv_python, str(script)]
    if mode == "full":
        cmd.append("--full-rebuild")

    log_path = INGEST_LOG_PATH

    try:
        with open(log_path, "w") as log_file:
            log_file.write(
                f"{datetime.now(timezone.utc).isoformat()} [INFO] "
                f"摄入任务启动 (mode={mode})\n"
            )
            _ingest_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
            )

        logger.info("摄入任务已启动: PID=%d, mode=%s", _ingest_process.pid, mode)

        return APIResponse.ok(IngestTriggerResponse(
            accepted=True,
            message=f"摄入任务已启动 (PID: {_ingest_process.pid})",
            mode=mode,
            pid=_ingest_process.pid,
        ))
    except Exception as e:
        logger.error("启动摄入任务失败: %s", e)
        raise HTTPException(status_code=500, detail=f"启动摄入失败: {e}")


@admin_router.get("/ingest/status", response_model=APIResponse[IngestStatus])
async def ingest_status(
    lines: int = 50,
) -> APIResponse[IngestStatus]:
    """查询摄入运行状态 + 日志尾."""
    global _ingest_process

    running = _ingest_process is not None and _ingest_process.poll() is None
    pid = _ingest_process.pid if _ingest_process else None

    # 读取日志尾
    log_tail: list[str] = []
    if INGEST_LOG_PATH.exists():
        try:
            all_lines = INGEST_LOG_PATH.read_text("utf-8").splitlines()
            log_tail = all_lines[-lines:]
        except Exception:
            pass

    # 最近摄入时间（从 manifest 获取）
    last_at = None
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text("utf-8"))
            times = [v.get("ingested_at", "") for v in data.get("specs", {}).values()]
            times = [t for t in times if t]
            if times:
                last_at = max(times)
        except Exception:
            pass

    return APIResponse.ok(IngestStatus(
        running=running,
        pid=pid,
        log_tail=log_tail,
        last_ingestion_at=last_at,
    ))


@admin_router.get("/manifest", response_model=APIResponse[list[ManifestItem]])
async def list_manifest() -> APIResponse[list[ManifestItem]]:
    """完整 manifest 清单."""
    if not MANIFEST_PATH.exists():
        return APIResponse.ok([], total=0)

    try:
        data = json.loads(MANIFEST_PATH.read_text("utf-8"))
        specs = data.get("specs", {})
        items = [
            ManifestItem(
                key=key,
                spec_number=v["spec_number"],
                release=v["release"],
                latest_version=v.get("latest_version", ""),
                file_path=v.get("file_path", ""),
                sha256=v.get("sha256", ""),
                chunk_count=v.get("chunk_count", 0),
                ingested_at=v.get("ingested_at", ""),
            )
            for key, v in specs.items()
        ]
        return APIResponse.ok(items, total=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 manifest 失败: {e}")


@admin_router.delete("/manifest/{key:path}", response_model=APIResponse[dict])
async def delete_manifest_record(key: str) -> APIResponse[dict]:
    """删除单条 manifest 记录."""
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=404, detail="Manifest 文件不存在")

    try:
        data = json.loads(MANIFEST_PATH.read_text("utf-8"))
        specs = data.get("specs", {})
        if key not in specs:
            raise HTTPException(status_code=404, detail=f"Key 不存在: {key}")

        removed = specs.pop(key)
        data["specs"] = specs
        MANIFEST_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

        return APIResponse.ok({
            "deleted_key": key,
            "spec_number": removed["spec_number"],
            "release": removed["release"],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


# ── 响应模型（新增） ──

class SystemInfo(BaseModel):
    """系统运行信息."""
    python_version: str
    platform: str
    uptime_seconds: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    milvus_connected: bool = False
    embedding_cache_entries: int = 0
    embedding_cache_mb: float = 0.0
    online_search_configured: bool = False


class LogEntry(BaseModel):
    """日志条目."""
    lines: list[str]
    total_lines: int
    level: str = "ALL"


class ConfigView(BaseModel):
    """非敏感配置视图."""
    llm_model: str = ""
    llm_base_url: str = ""
    embedding_device: str = ""
    embedding_provider: str = ""
    chunk_size: int = 0
    chunk_overlap: int = 0
    dense_top_k: int = 0
    bm25_top_k: int = 0
    milvus_host: str = ""
    milvus_port: int = 0
    online_search_enabled: bool = False
    reranker_enabled_by_default: bool = False


# ── 端点（新增） ──

@admin_router.get("/logs", response_model=APIResponse[LogEntry])
async def system_logs(
    level: str = "ALL",
    lines: int = 100,
) -> APIResponse[LogEntry]:
    """读取应用日志尾行，支持按级别过滤."""
    log_path = settings.project_root / "logs" / "app.log"
    if not log_path.exists():
        return APIResponse.ok(LogEntry(lines=[], total_lines=0, level=level))

    try:
        all_lines = log_path.read_text("utf-8").splitlines()
        total = len(all_lines)

        # 级别过滤
        if level != "ALL":
            level_upper = level.upper()
            all_lines = [l for l in all_lines if level_upper in l]

        # 取尾行
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return APIResponse.ok(LogEntry(lines=tail, total_lines=total, level=level))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {e}")


@admin_router.get("/system", response_model=APIResponse[SystemInfo])
async def system_info() -> APIResponse[SystemInfo]:
    """系统运行信息：内存、磁盘、Milvus 状态."""
    # 内存 (macOS/Linux 通用)
    try:
        mem_info = _get_memory_info()
    except Exception:
        mem_info = {"used_mb": 0, "total_mb": 0, "percent": 0}

    # 磁盘
    try:
        usage = shutil.disk_usage(str(settings.project_root))
        disk_used_gb = round(usage.used / (1024 ** 3), 1)
        disk_total_gb = round(usage.total / (1024 ** 3), 1)
        disk_percent = round((usage.used / usage.total) * 100, 1)
    except Exception:
        disk_used_gb = disk_total_gb = disk_percent = 0

    # Milvus 连接
    milvus_ok = False
    try:
        pipeline = get_pipeline()
        store = pipeline._store
        milvus_ok = getattr(store, "_collection", None) is not None
    except Exception:
        pass

    # Uptime
    uptime = time.time() - _start_time

    return APIResponse.ok(SystemInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        uptime_seconds=round(uptime, 0),
        memory_used_mb=round(mem_info["used_mb"], 1),
        memory_total_mb=round(mem_info["total_mb"], 1),
        memory_percent=round(mem_info["percent"], 1),
        disk_used_gb=disk_used_gb,
        disk_total_gb=disk_total_gb,
        disk_percent=disk_percent,
        milvus_connected=milvus_ok,
        embedding_cache_entries=_get_cache_stats()[0],
        embedding_cache_mb=_get_cache_stats()[1],
        online_search_configured=_check_online_search(),
    ))


@admin_router.get("/config", response_model=APIResponse[ConfigView])
async def view_config() -> APIResponse[ConfigView]:
    """查看非敏感配置."""
    return APIResponse.ok(ConfigView(
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        embedding_device=settings.embedding_device,
        embedding_provider=settings.embedding_provider,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port,
        online_search_enabled=getattr(settings, 'enable_online_search', False),
        reranker_enabled_by_default=getattr(settings, 'reranker_enabled', True),
    ))


# ── 辅助函数 ──

def _build_doc_map(pipeline) -> dict[str, dict[str, Any]]:
    """从向量库构建文档摘要 map."""
    store = pipeline._store

    if hasattr(store, "get_documents_summary"):
        return store.get_documents_summary()

    # Fallback: 空 map
    return {}


def _get_memory_info() -> dict[str, float]:
    """跨平台内存信息 (macOS / Linux). 无需 psutil."""
    try:
        import subprocess as sp
        if sys.platform == "darwin":
            vm = sp.check_output(["vm_stat"], text=True)
            page_size = int(sp.check_output(["sysctl", "-n", "hw.pagesize"]).strip())
            # 用 hw.memsize 获取物理内存总量 (Apple Silicon 统一内存准确)
            total_mb = int(sp.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / (1024 ** 2)
            lines: dict[str, int] = {}
            for line in vm.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    key = k.strip().strip('"')
                    val = v.strip().rstrip(".")
                    # macOS 26: 首行含 "(page size of ...)" 无法转为 int
                    try:
                        lines[key] = int(val)
                    except ValueError:
                        continue
            # Activity Monitor 接近公式: Anonymous + Wired + Compressed
            # (仅用 Active+Wired+Compressed 会漏掉大量匿名非活跃页, 低估 6-7GB)
            anon_pages = lines.get("Anonymous pages", 0)
            used_pages = (
                anon_pages +
                lines.get("Pages wired down", 0) +
                lines.get("Pages occupied by compressor", 0)
            )
            used_mb = (used_pages * page_size) / (1024 ** 2)
            return {"used_mb": used_mb, "total_mb": total_mb, "percent": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0}
        else:
            with open("/proc/meminfo") as f:
                mem = f.read()
            def _val(k):
                for line in mem.splitlines():
                    if line.startswith(k):
                        return int(line.split()[1]) / 1024
                return 0
            total_mb = _val("MemTotal:")
            avail_mb = _val("MemAvailable:")
            used_mb = total_mb - avail_mb
            return {"used_mb": max(used_mb, 1), "total_mb": total_mb, "percent": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0}
    except Exception:
        return {"used_mb": 0, "total_mb": 0, "percent": 0}


def _get_cache_stats() -> tuple[int, float]:
    """获取嵌入缓存统计 (entries, size_mb)."""
    try:
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache()
        st = cache.stats()
        return st["total_entries"], st["size_mb"]
    except Exception:
        return 0, 0.0


def _check_online_search() -> bool:
    """检查在线搜索是否已配置 (至少一个数据源可用)."""
    google_ok = bool(getattr(settings, 'google_api_key', '') and getattr(settings, 'google_cse_id', ''))
    tspec_ok = bool(getattr(settings, 'tspec_llm_url', ''))
    return getattr(settings, 'enable_online_search', False) and (google_ok or tspec_ok)
