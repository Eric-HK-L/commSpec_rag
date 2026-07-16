"""摄入编排器 — 串联 下载→转换→分块→嵌入→入库 全流程.

支持:
  - 一键全流程: run_full_pipeline()
  - 分段执行: --skip-* 跳过已完成步骤
  - 中间结果落盘: data/interim/
  - CLI 集成: 扩展 src/cli.py ingest 子命令
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import settings
from src.ingestion.embedder import BatchEmbedder
from src.ingestion.embedding_cache import EmbeddingCache
from src.ingestion.extractor import DoclingExtractor, ExtractionResult
from src.ingestion.splitter import HeaderAwareSplitter
from src.retriever.vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """摄入统计."""
    docs_total: int = 0
    docs_success: int = 0
    chunks_total: int = 0
    chunks_inserted: int = 0
    cache_hits: int = 0
    cache_total: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class IngestionOrchestrator:
    """摄入全流程编排器."""

    def __init__(
        self,
        vector_store: VectorStore,
        interim_dir: str | None = None,
        skip_download: bool = False,
        skip_extract: bool = False,
        skip_split: bool = False,
        skip_embed: bool = False,
        on_step: Callable[[str, str], None] | None = None,
    ):
        self._store = vector_store
        self._interim = Path(interim_dir) if interim_dir else settings.data_abs_dir / "interim"
        self._skip = {
            "download": skip_download,
            "extract": skip_extract,
            "split": skip_split,
            "embed": skip_embed,
        }
        self._on_step = on_step

        self._splitter = HeaderAwareSplitter(
            max_chunk_chars=getattr(settings, "chunk_size", 2500),
            chunk_overlap_chars=getattr(settings, "chunk_overlap", 100),
        )

    # ── 全流程 ──

    def run_full_pipeline(
        self,
        release: str = "R18",
        series: int | None = None,
        spec: str | None = None,
        docx_dir: str | None = None,
    ) -> IngestionStats:
        """执行完整摄入管线.

        Args:
            release: 3GPP Release.
            series: Series 编号.
            spec: 单篇规范号.
            docx_dir: 直接指定 DOCX 目录 (跳过下载).

        Returns:
            IngestionStats 含全流程统计.
        """
        stats = IngestionStats()
        t_start = time.time()

        # Step 1: 下载
        if not self._skip["download"] and docx_dir is None:
            self._notify("download", "running")
            from scripts.download_specs import SpecDownloader
            downloader = SpecDownloader(output_dir=str(settings.documents_abs_dir))
            downloader.download(release=release, series=series, spec=spec, dry_run=False)
            self._notify("download", "completed")

        # Step 2: 查找 DOCX
        search_dir = Path(docx_dir) if docx_dir else settings.documents_abs_dir
        docx_files = list(search_dir.rglob("*.docx"))
        stats.docs_total = len(docx_files)
        logger.info("找到 %d 个 DOCX 文件", stats.docs_total)

        if not docx_files:
            stats.errors.append("未找到 DOCX 文件")
            return stats

        # Step 3: 提取 Markdown
        extraction_results: list[ExtractionResult] = []
        if not self._skip["extract"]:
            self._notify("extract", "running")
            extractor = DoclingExtractor()
            extraction_results = extractor.extract_directory(str(search_dir))
            stats.docs_success = sum(1 for r in extraction_results if r.markdown)
            self._notify("extract", "completed")
        else:
            # 从 interim 加载已提取的 MD
            extraction_results = self._load_interim_extractions(search_dir.name)

        # Step 4: 分块
        if not self._skip["split"]:
            self._notify("split", "running")
            chunks = self._split_all(extraction_results)
            stats.chunks_total = len(chunks)
            # 保存到 interim
            self._save_chunks_interim(search_dir.name, chunks)
            self._notify("split", "completed")
        else:
            chunks = self._load_chunks_interim(search_dir.name)
            stats.chunks_total = len(chunks)

        # Step 5: 嵌入 (含 SQLite 双层缓存)
        if not self._skip["embed"] and chunks:
            self._notify("embed", "running")
            sqlite_cache = EmbeddingCache()
            cache_stats_before = sqlite_cache.stats()
            embedder = BatchEmbedder(
                cache_dir=str(settings.data_abs_dir / "cache" / "embeddings"),
                sqlite_cache=sqlite_cache,
            )
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_batch(texts)

            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            cache_stats_after = sqlite_cache.stats()
            stats.cache_hits = cache_stats_after["total_entries"] - cache_stats_before["total_entries"]
            stats.cache_total = cache_stats_after["total_entries"]
            logger.info(
                "嵌入缓存: 新增 %d 条, 总计 %d 条 (%.1f MB)",
                stats.cache_hits, stats.cache_total, cache_stats_after["size_mb"],
            )
            self._notify("embed", "completed")

        # Step 6: 入库
        if chunks:
            self._notify("insert", "running")
            stats.chunks_inserted = self._store.insert(chunks)
            self._notify("insert", "completed")

        stats.elapsed_seconds = time.time() - t_start
        logger.info("摄入完成: %d docs, %d chunks, %.1fs", stats.docs_success, stats.chunks_inserted, stats.elapsed_seconds)
        return stats

    # ── 内部方法 ──

    def _split_all(self, results: list[ExtractionResult]) -> list[Chunk]:
        """对所有提取结果分块."""
        all_chunks: list[Chunk] = []
        for r in results:
            if not r.markdown:
                continue
            doc_meta = {
                "doc_id": Path(r.source_file).stem,
                "series": int(r.spec_number.split(".")[0]) if r.spec_number else 0,
                "spec_number": r.spec_number,
                "release": r.release,
            }
            chunks = self._splitter.split_document(r.markdown, doc_meta)
            all_chunks.extend(chunks)
        return all_chunks

    def _save_chunks_interim(self, key: str, chunks: list[Chunk]) -> None:
        """保存 chunk 中间结果到 JSON."""
        self._interim.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "text": c.text,
                "doc_id": c.doc_id,
                "series": c.series,
                "spec_number": c.spec_number,
                "release": c.release,
                "parent_section_id": c.parent_section_id,
                "parent_title": c.parent_title,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        path = self._interim / f"{key}_chunks.json"
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("保存 %d chunks → %s", len(chunks), path)

    def _load_chunks_interim(self, key: str) -> list[Chunk]:
        """从 JSON 加载中间 chunk 结果."""
        path = self._interim / f"{key}_chunks.json"
        with open(path) as f:
            data = json.load(f)
        return [
            Chunk(
                text=d["text"], doc_id=d["doc_id"], series=d["series"],
                spec_number=d["spec_number"], release=d["release"],
                parent_section_id=d["parent_section_id"],
                parent_title=d["parent_title"],
                chunk_index=d["chunk_index"],
                embedding=None,
            )
            for d in data
        ]

    def _load_interim_extractions(self, key: str) -> list[ExtractionResult]:
        """从 interim 加载提取结果 (简化版, 仅恢复 markdown)."""
        path = self._interim / f"{key}_extractions.json"
        if not path.exists():
            logger.warning("interim 文件不存在: %s", path)
            return []
        with open(path) as f:
            data = json.load(f)
        return [
            ExtractionResult(
                source_file=d["source_file"],
                spec_number=d.get("spec_number", ""),
                release=d.get("release", ""),
                version=d.get("version", ""),
                title=d.get("title", ""),
                markdown=d.get("markdown", ""),
            )
            for d in data
        ]

    def _notify(self, step: str, status: str) -> None:
        if self._on_step:
            self._on_step(step, status)
