"""Phase 4 管理 API — 知识库运维端点."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
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


# ── 辅助函数 ──

def _build_doc_map(pipeline) -> dict[str, dict[str, Any]]:
    """从向量库构建文档摘要 map."""
    store = pipeline._store

    if hasattr(store, "get_documents_summary"):
        return store.get_documents_summary()

    # Fallback: 空 map
    return {}
