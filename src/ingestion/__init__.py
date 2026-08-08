"""文档摄入管线 — 规范 ETL 全流程.

Extractor          — DOCX→Markdown (Pandoc)
HeaderAwareSplitter — 标题感知分块
BatchEmbedder       — 批量嵌入生成
IngestionOrchestrator — 全流程编排
IncrementalIndexer  — 增量索引

.. deprecated:: 2026-07-14
    PrecomputedLoader 已废弃，不再使用 HuggingFace 预计算数据集。
"""

from .embedder import BatchEmbedder
from .extractor import ExtractionResult, PandocExtractor
from .incremental import IncrementalIndexer, IndexEntry
from .orchestrator import IngestionOrchestrator, IngestionStats
from .splitter import HeaderAwareSplitter

__all__ = [
    "PandocExtractor",
    "ExtractionResult",
    "HeaderAwareSplitter",
    "BatchEmbedder",
    "IngestionOrchestrator",
    "IngestionStats",
    "IncrementalIndexer",
    "IndexEntry",
]
