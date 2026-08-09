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
    section_number: str = ""    # chunk 自身的章节编号，如 "7.1.1"
    section_title: str = ""     # 章节标题，如 "UE behaviour"
    section_path: str = ""      # 层级路径
    doc_type: str = "3gpp"      # 文档类型: "3gpp" | "oran"
    # chunk 元数据（摄入时写入 Milvus，检索时零推导开销）
    content_type: str = ""       # "parameter_table" | "definition" | "procedure" | "overview"
    spec_role: str = ""          # "authoritative" | "supporting" | "overview"
    topic_domain: str = ""       # "phy_layer" | "mac_layer" | "rrc_layer" | "ran_arch"
    # small-to-big 父上下文 (来自 Milvus parent_text / parent_chunk_id 字段)
    parent_text: str = ""
    parent_chunk_id: int = 0
    # 扩展上下文：同文档相邻 chunk
    adjacent_chunks: list[str] = field(default_factory=list)

    @classmethod
    def from_search_result(cls, sr: SearchResult) -> "RetrievalResult":
        result = cls(
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
            section_number=sr.section_number,
            section_title=sr.section_title,
            section_path=sr.section_path,
            doc_type=sr.doc_type,
            content_type=sr.content_type,
            spec_role=sr.spec_role,
            topic_domain=sr.topic_domain,
            parent_text=getattr(sr, "parent_text", ""),
            parent_chunk_id=getattr(sr, "parent_chunk_id", 0),
        )
        src_tag = getattr(sr, "_source_tag", None)
        if src_tag:
            result._source_tag = src_tag
        return result

    def to_context_str(self, index: int = 0) -> str:
        """格式化为 LLM 上下文字符串."""
        parts = []
        # 元数据标签（优先展示权威性标记）
        role_map = {"authoritative": "🔴权威定义", "supporting": "🟡补充参考", "overview": "⚪概述"}
        type_map = {"parameter_table": "📊参数表", "definition": "📋定义"}
        if self.spec_role and self.spec_role in role_map:
            parts.append(role_map[self.spec_role])
        if self.content_type and self.content_type in type_map:
            parts.append(type_map[self.content_type])
        if self.spec_number:
            parts.append(f"TS {self.spec_number}")
        section_ref = self.section_number or self.parent_section_id
        if section_ref:
            parts.append(f"§{section_ref}")
        if self.parent_title:
            parts.append(self.parent_title)
        if self.release:
            parts.append(f"({self.release})")

        header = " | ".join(parts) if parts else f"Doc: {self.doc_id}"
        body = f"[{header}]\n{self.text}"

        # 附加相邻 chunk 上下文 (最多 4 条)
        if self.adjacent_chunks:
            adj_text = "\n".join(
                f"  [{i}] {t[:500]}"
                for i, t in enumerate(self.adjacent_chunks[:4])
            )
            body += f"\n\n[相邻上下文]\n{adj_text}"

        return body


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
        filter_expr: str | None = None,
        final_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """执行混合检索.

        Args:
            query: 原始查询文本.
            query_embedding: 查询的向量嵌入 (1024-dim float32).
            filter_expr: Milvus 标量过滤表达式, 如 'release == "R18" && doc_type == "3gpp"'.
            final_top_k: 返回条数覆盖值 (None 用实例默认), 避免调用方临时改共享状态.

        Returns:
            按 RRF 融合排序的检索结果.
        """
        k = self._final_top_k if final_top_k is None else final_top_k
        if self._store.supports_bm25 and hasattr(self._store, "hybrid_search"):
            # Milvus 原生混合检索
            results = self._store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                dense_top_k=self._dense_top_k,
                sparse_top_k=self._sparse_top_k,
                final_top_k=k,
                filter_expr=filter_expr,
            )
        else:
            # 降级：仅 Dense 检索
            logger.warning("向量库不支持 BM25，降级为纯 Dense 检索")
            results = self._store.search_dense(
                query_embedding=query_embedding,
                top_k=k,
                filter_expr=filter_expr,
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
