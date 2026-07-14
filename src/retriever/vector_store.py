"""向量存储抽象层 — Milvus 主后端."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """单条检索结果."""

    chunk_id: str | int
    text: str
    score: float
    doc_id: str = ""
    series: int = 0
    spec_number: str = ""
    release: str = ""
    parent_section_id: str = ""
    parent_title: str = ""
    chunk_index: int = 0

    def to_context_str(self, index: int = 0) -> str:
        """将检索结果格式化为注入 LLM 上下文的字符串."""
        header = f"[来源: 3GPP TS {self.spec_number} (Series {self.series}), §{self.parent_section_id}]"
        return f"{header}\n{self.text}"


@dataclass
class Chunk:
    """待入库的文档片段."""

    text: str
    embedding: np.ndarray | None = None
    doc_id: str = ""
    series: int = 0
    spec_number: str = ""
    release: str = ""
    parent_section_id: str = ""
    parent_title: str = ""
    chunk_index: int = 0


# ═══════════════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════════════

class VectorStore(ABC):
    """向量数据库统一接口."""

    @abstractmethod
    def connect(self) -> None:
        """建立连接."""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接."""

    @abstractmethod
    def create_collection(self, drop_existing: bool = False) -> None:
        """创建集合/索引."""

    @abstractmethod
    def insert(self, chunks: list[Chunk]) -> int:
        """批量插入 chunk，返回插入数量."""

    @abstractmethod
    def search_dense(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> list[SearchResult]:
        """Dense 向量检索."""

    @abstractmethod
    def search_sparse(
        self, query_text: str, top_k: int = 100
    ) -> list[SearchResult]:
        """BM25 稀疏检索."""

    @abstractmethod
    def delete_by_filter(self, filter_expr: str) -> int:
        """按过滤条件删除."""

    @property
    @abstractmethod
    def count(self) -> int:
        """当前集合中的记录数."""

    @property
    @abstractmethod
    def supports_bm25(self) -> bool:
        """是否原生支持 BM25 稀疏检索."""
