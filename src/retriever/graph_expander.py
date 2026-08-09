"""图增强检索 — 离线 Xref Graph 邻接查询，替换在线二次检索。

在 RAG 检索命中 chunk 后，沿 REFERENCES 边确定性发现跨规范关联 chunk，
不依赖向量 recall，延迟极低 (O(1) 邻接查询)。

使用方式:
    expander = GraphExpander("data/processed/xref_graph.json", milvus_store)
    expanded = expander.expand(results, max_per_chunk=5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class GraphExpander:
    """离线 Xref Graph 加载器 + 邻接查询扩展。

    从预构建的 xref_graph.json 加载节点/边/邻接索引，
    对检索结果做图增强扩展。
    """

    def __init__(
        self,
        graph_path: str | Path,
        store=None,
    ):
        """初始化图扩展器。

        Args:
            graph_path: xref_graph.json 文件路径。
            store: MilvusStore 实例，用于加载扩展 chunk 的完整 content。
                   为 None 时可在 expand 时传入。
        """
        self._graph_path = Path(graph_path)
        self._store = store
        self._graph: dict | None = None
        self._adjacency: dict[str, dict[str, list[str]]] = {}
        self._node_map: dict[str, dict] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """图是否已加载。"""
        return self._loaded

    def load(self) -> bool:
        """加载 xref_graph.json。

        Returns:
            True 表示加载成功。
        """
        if self._loaded:
            return True

        if not self._graph_path.exists():
            logger.warning("Xref Graph 文件不存在: %s", self._graph_path)
            return False

        try:
            with open(self._graph_path, encoding="utf-8") as f:
                self._graph = json.load(f)

            self._adjacency = self._graph.get("adjacency", {})
            self._node_map = {n["id"]: n for n in self._graph.get("nodes", [])}
            self._loaded = True

            meta = self._graph.get("metadata", {})
            logger.info(
                "Xref Graph 已加载: %d 节点, %d 边 (%s)",
                meta.get("total_nodes", 0),
                meta.get("total_edges", 0),
                self._graph_path,
            )
            return True
        except Exception as e:
            logger.error("加载 Xref Graph 失败: %s", e)
            return False

    def expand(
        self,
        results: list,
        max_per_chunk: int = 5,
        top_n: int = 10,
        cross_spec_only: bool = True,
        store=None,
    ) -> list:
        """沿 REFERENCES 边扩展检索结果中的跨规范关联 chunk。

        Args:
            results: 检索结果列表 (RetrievalResult / SearchResult)。
            max_per_chunk: 每个命中 chunk 最多扩展条数。
            top_n: 仅对前 N 条命中 chunk 做扩展 (控制延迟)。
            cross_spec_only: 仅保留跨规范新发现 (默认 True)。
            store: 可选的 MilvusStore (未在 __init__ 传入时使用)。

        Returns:
            新发现的 chunk 列表 (SearchResult / RetrievalResult 格式)。
        """
        if not self._loaded and not self.load():
            return []

        store = store or self._store
        if store is None:
            logger.warning("GraphExpander 未配置 store, 无法加载扩展 chunk")
            return []

        # 获取已命中 chunk 的 id 和 spec 信息
        seen_ids: set[str] = set()
        seen_specs: set[str] = set()
        for r in results:
            cid = str(getattr(r, "chunk_id", ""))
            if cid:
                seen_ids.add(cid)
            spec = getattr(r, "spec_number", "")
            if spec:
                seen_specs.add(spec)

        expanded: list = []
        for r in results[:top_n]:
            cid = str(getattr(r, "chunk_id", ""))
            if not cid:
                continue

            # 查邻接表
            adj = self._adjacency.get(cid, {})
            ref_ids = adj.get("REFERENCES", [])[:max_per_chunk]

            ref_spec = getattr(r, "spec_number", "")
            for ref_id in ref_ids:
                if ref_id in seen_ids:
                    continue

                node = self._node_map.get(ref_id, {})
                target_spec = node.get("spec", "")

                # cross_spec_only: 跳过同规范
                if cross_spec_only and target_spec == ref_spec:
                    continue

                seen_ids.add(ref_id)
                if target_spec:
                    seen_specs.add(target_spec)

                # 加载 chunk 的完整 content
                # rank-based 分数 (与 multi_hop 的 RRF k=60 同尺度):
                # 让补充 chunk 可排序、可被下游评分, 而非硬编码 0.0
                chunk = self._load_chunk_content(
                    int(ref_id), store, score=self._rank_score(len(expanded) + 1),
                )
                if chunk:
                    chunk._source_tag = "graph_expand"
                    expanded.append(chunk)

        if expanded:
            logger.info(
                "Graph Expand: %d 条新增 cross-spec chunk (top-%d hit, max %d each)",
                len(expanded), top_n, max_per_chunk,
            )
        return expanded

    @staticmethod
    def _rank_score(rank: int, k: int = 60) -> float:
        """rank-based 分数, 与 RRF 融合 (milvus_store._rrf_fuse, k=60) 同尺度."""
        return 1.0 / (k + rank)

    def _load_chunk_content(self, chunk_id: int, store, score: float) -> "SearchResult | None":
        """从 Milvus 根据 chunk_id 加载完整文本。

        注意：Milvus auto_id 生成的 id 为 INT64，graph 中的 id 也是 int，
        但由于 Milvus query 需要 expr，需用 id == X 精确查询。
        """
        try:
            results = store._collection.query(
                expr=f"id == {chunk_id}",
                output_fields=[
                    "text", "doc_id", "series", "spec_number", "release",
                    "parent_section_id", "parent_title", "chunk_index",
                    "section_number", "section_title", "section_path",
                ],
                limit=1,
            )
            if not results:
                return None

            from src.retriever.vector_store import SearchResult

            r = results[0]
            return SearchResult(
                chunk_id=r.get("id", chunk_id),
                text=r.get("text", ""),
                score=score,  # 图扩展 chunk 无向量分数, 用 rank-based 分数排序
                doc_id=r.get("doc_id", ""),
                series=r.get("series", 0),
                spec_number=r.get("spec_number", ""),
                release=r.get("release", ""),
                parent_section_id=r.get("parent_section_id", ""),
                parent_title=r.get("parent_title", ""),
                chunk_index=r.get("chunk_index", 0),
                section_number=r.get("section_number", ""),
                section_title=r.get("section_title", ""),
                section_path=r.get("section_path", ""),
            )
        except Exception as e:
            logger.debug("加载 chunk %d 失败: %s", chunk_id, e)
            return None
