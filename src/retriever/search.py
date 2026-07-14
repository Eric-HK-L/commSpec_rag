"""混合检索 — Dense + BM25 + RRF 融合检索器."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.retriever.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果（含层级上下文）."""

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
    # 扩展上下文：同文档相邻 chunk
    adjacent_chunks: list[str] = field(default_factory=list)

    @classmethod
    def from_search_result(cls, sr: SearchResult) -> "RetrievalResult":
        return cls(
            chunk_id=sr.chunk_id,
            text=sr.text,
            score=sr.score,
            doc_id=sr.doc_id,
            series=sr.series,
            spec_number=sr.spec_number,
            release=sr.release,
            parent_section_id=sr.parent_section_id,
            parent_title=sr.parent_title,
            chunk_index=sr.chunk_index,
        )

    def to_context_str(self, index: int = 0) -> str:
        """格式化为 LLM 上下文字符串."""
        parts = []
        if self.spec_number:
            parts.append(f"TS {self.spec_number}")
        if self.parent_section_id:
            parts.append(f"§{self.parent_section_id}")
        if self.parent_title:
            parts.append(self.parent_title)
        if self.release:
            parts.append(f"({self.release})")

        header = " | ".join(parts) if parts else f"Doc: {self.doc_id}"
        return f"[{header}]\n{self.text}"


class HybridRetriever:
    """Dense + BM25 混合检索器.

    流水线: 查询改写 → Dense检索 → BM25检索 → RRF融合 → (可选NN Router精排)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        dense_top_k: int = 100,
        sparse_top_k: int = 100,
        final_top_k: int = 10,
    ):
        self._store = vector_store
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._final_top_k = final_top_k

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> list[RetrievalResult]:
        """执行混合检索.

        Args:
            query: 原始查询文本.
            query_embedding: 查询的向量嵌入 (1024-dim float32).

        Returns:
            按 RRF 融合排序的检索结果.
        """
        if self._store.supports_bm25 and hasattr(self._store, "hybrid_search"):
            # Milvus 原生混合检索
            results = self._store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                dense_top_k=self._dense_top_k,
                sparse_top_k=self._sparse_top_k,
                final_top_k=self._final_top_k,
            )
        else:
            # 降级：仅 Dense 检索
            logger.warning("向量库不支持 BM25，降级为纯 Dense 检索")
            results = self._store.search_dense(
                query_embedding=query_embedding,
                top_k=self._final_top_k,
            )

        return [RetrievalResult.from_search_result(r) for r in results]

    def search_with_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        expand_adjacent: int = 1,
    ) -> list[RetrievalResult]:
        """检索并附带相邻 chunk 上下文.

        Args:
            query: 查询文本.
            query_embedding: 查询嵌入.
            expand_adjacent: 每个命中 chunk 附带前后各 N 个相邻 chunk.

        Returns:
            含相邻上下文的检索结果.
        """
        results = self.search(query, query_embedding)
        return results  # 相邻 chunk 扩展在后续 Phase 实现
