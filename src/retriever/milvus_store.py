"""Milvus 2.4+ 向量存储后端 — Dense 检索 + Python BM25 混合检索."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# ── BGE-M3 1024-dim 向量批量 insert 可能超 gRPC 64MB 默认上限, 必须在 pymilvus import 前设置 ──
os.environ["GRPC_DEFAULT_MAX_RECEIVE_MESSAGE_LENGTH"] = str(256 * 1024 * 1024)
os.environ["GRPC_DEFAULT_MAX_SEND_MESSAGE_LENGTH"] = str(256 * 1024 * 1024)

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from src.config import settings

from .bm25_index import BM25Indexer
from .vector_store import Chunk, SearchResult, VectorStore

logger = logging.getLogger(__name__)


def _matches_filter_expr(meta: dict, filter_expr: str) -> bool:
    """用 Milvus 标量过滤表达式匹配单条 chunk 元数据.

    支持 _build_filter_expr 生成的子集:
      - 原子: `field == "value"` / `field == 38`
      - 组合: `&&` (AND), `||` (OR)
    用于 BM25 结果在 Python 侧做同样的过滤.
    """
    if not filter_expr:
        return True
    if not meta:
        return False

    def _atom(expr: str) -> bool:
        expr = expr.strip().strip("()")
        if "==" not in expr:
            return False
        field, _, raw = expr.partition("==")
        field = field.strip()
        value = raw.strip().strip('"')
        actual = meta.get(field)
        # Milvus 数值字段 (series) 以字符串形式写入 meta 时统一比较
        return str(actual) == str(value) or actual == value

    # 拆分为 OR 组, 每组内 AND
    for group in filter_expr.split("||"):
        atoms = group.split("&&")
        if all(_atom(a) for a in atoms if a.strip()):
            return True
    return False

# Milvus 字段配置
VARCHAR_MAX = 65000  # 字节级安全边距 (Milvus VARCHAR 硬限 65535 bytes)


def _safe_truncate_bytes(text: str, max_bytes: int) -> str:
    """按字节数智能截断 — 在语义边界切断, 保留完整语义单元。

    3GPP 文档场景: 正常流程 splitter 已保证 ≤ max_chunk_bytes (55KB),
    本函数仅作为 Milvus VARCHAR 65535 硬限制的最后兜底。

    截断优先级: 段落 → 行 → 句 → 分句 → 词 → 字符
    若截断发生, 附加 "…" 标记并记录 warning 日志。
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    # 预留 "…" 标记 3 字节
    limit = max_bytes - 3
    truncated = encoded[:limit].decode("utf-8", errors="ignore")

    # 语义边界优先级: (分隔符, 最早可接受位置比例)
    candidates: list[tuple[str, float]] = [
        ("\n\n", 0.3),   # 段落末尾 — 最优
        ("\n", 0.4),     # 行末
        (". ", 0.5),     # 英文句末
        ("。", 0.5),     # 中文句末
        ("; ", 0.6),     # 分句
        ("\n", 0.2),     # 行末 (放宽位置限制)
        (" ", 0.7),      # 词边界
    ]

    for sep, min_ratio in candidates:
        pos = truncated.rfind(sep)
        if pos >= int(len(truncated) * min_ratio):
            result = truncated[:pos].rstrip() + "…"
            logger.warning(
                "文本截断: %dB → %dB (在 %r 边界, 丢失 %.0f%%)",
                len(encoded),
                len(result.encode("utf-8")),
                sep,
                (1 - len(result.encode("utf-8")) / len(encoded)) * 100,
            )
            return result

    # 最后手段: 干净 UTF-8 切断
    logger.warning("文本硬截断: %dB → %dB (无合适语义边界)", len(encoded), max_bytes)
    return truncated + "…"


def _escape_milvus_expr(value: str) -> str:
    """转义 Milvus 表达式中的单引号，防止查询注入。"""
    return value.replace("'", "''")


class MilvusStore(VectorStore):
    """Milvus 2.4+ 向量数据库 — Dense + Python BM25 混合检索."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "TeleComm_specs",
        alias: str = "default",
        bm25_index_path: str | Path | None = None,
    ):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._alias = alias
        self._collection: Collection | None = None
        self._connected = False
        self._bm25 = BM25Indexer(bm25_index_path) if bm25_index_path else BM25Indexer()

    # ── 连接管理 ──

    def connect(self) -> None:
        """连接到 Milvus 服务."""
        if self._connected:
            return
        try:
            connections.connect(
                alias=self._alias,
                host=self._host,
                port=self._port,
                grpc_max_receive_message_length=256 * 1024 * 1024,
                grpc_max_send_message_length=256 * 1024 * 1024,
            )
            self._connected = True
            logger.info("Milvus 连接成功: %s:%d", self._host, self._port)

            # 自动加载 BM25 索引
            if self._bm25.load():
                logger.info("BM25 索引已加载: %d 条", self._bm25.doc_count)
            else:
                logger.warning("BM25 索引未找到或加载失败, 将降级为纯 Dense 检索")
        except MilvusException as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    def disconnect(self) -> None:
        """断开 Milvus 连接."""
        if self._connected:
            connections.disconnect(self._alias)
            self._connected = False
            self._collection = None

    # ── 集合管理 ──

    def create_collection(self, drop_existing: bool = False) -> None:
        """创建 3GPP 规范检索集合."""
        self._ensure_connected()

        if drop_existing and utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
            logger.info("已删除旧集合: %s", self._collection_name)

        if utility.has_collection(self._collection_name):
            logger.info("集合已存在: %s", self._collection_name)
            self._collection = Collection(self._collection_name)
            self._collection.load()
            return

        # 定义字段 (Dense-only, BM25 待后续启用)
        # section_path/parent_title 用 4096: marked 数据集 O-RAN 嵌套深 + 中文 UTF-8 字节数 > 字符数
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=VARCHAR_MAX),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.embedding_dimension,
            ),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="series", dtype=DataType.INT64),
            FieldSchema(name="spec_number", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="release", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="parent_section_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="parent_title", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="section_number", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="spec_role", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="topic_domain", dtype=DataType.VARCHAR, max_length=32),
        ]

        schema = CollectionSchema(fields, description="3GPP 规范检索集合 (Dense)")
        self._collection = Collection(self._collection_name, schema)

        # 创建 Dense 向量索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        }
        self._collection.create_index(
            field_name="dense_vector", index_params=index_params
        )

        self._collection.load()
        logger.info(
            "集合创建成功: %s (dense=%ddim)", self._collection_name, settings.embedding_dimension
        )

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()
        if self._collection is None and utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()  # 必须 load, 否则 expr 标量过滤返回空

    # ── 数据操作 ──

    # BGE-M3 1024-dim 向量 + 3GPP 长文本 (ASN.1 可达 60KB/chunk):
    # 5000 chunks 最坏消息体 ≈ 332 MB > gRPC 任何合理上限
    # → 内部按 MAX_INSERT_BATCH 微批次插入, 根治 gRPC 超限
    MAX_INSERT_BATCH = 1000

    def insert(self, chunks: list[Chunk]) -> int:
        """批量插入文档 chunk (内部微批次, 防止 gRPC 消息超限)."""
        self._ensure_connected()
        if not chunks:
            return 0

        if self._collection is None:
            raise RuntimeError("集合未初始化，请先调用 create_collection()")

        total_inserted = 0

        for batch_start in range(0, len(chunks), self.MAX_INSERT_BATCH):
            batch = chunks[batch_start : batch_start + self.MAX_INSERT_BATCH]
            try:
                n = self._insert_batch(batch)
                total_inserted += n
            except MilvusException:
                logger.error(
                    "微批次插入失败 (offset=%d, size=%d), 已插入 %d 条",
                    batch_start, len(batch), total_inserted,
                )
                raise

        return total_inserted

    def _insert_batch(self, chunks: list[Chunk]) -> int:
        """单次插入 (≤ MAX_INSERT_BATCH chunks)."""
        data: list[list[Any]] = [
            [],  # text
            [],  # dense_vector
            [],  # doc_id
            [],  # series
            [],  # spec_number
            [],  # release
            [],  # parent_section_id
            [],  # parent_title
            [],  # chunk_index
            [],  # section_number
            [],  # section_title
            [],  # section_path
            [],  # doc_type
            [],  # content_type
            [],  # spec_role
            [],  # topic_domain
        ]

        for c in chunks:
            data[0].append(_safe_truncate_bytes(c.text, VARCHAR_MAX))
            vec = c.embedding if c.embedding is not None else np.zeros(settings.embedding_dimension, dtype=np.float32)
            data[1].append(vec.astype(np.float32).tolist())
            data[2].append(c.doc_id[:256] if c.doc_id else "")
            data[3].append(c.series)
            data[4].append(c.spec_number[:64] if c.spec_number else "")
            data[5].append(c.release[:32] if c.release else "")
            data[6].append(c.parent_section_id[:256] if c.parent_section_id else "")
            data[7].append(_safe_truncate_bytes(c.parent_title, 4096) if c.parent_title else "")
            data[8].append(c.chunk_index)
            data[9].append(c.section_number[:64] if c.section_number else "")
            data[10].append(c.section_title[:512] if c.section_title else "")
            data[11].append(_safe_truncate_bytes(c.section_path, 4096) if c.section_path else "")
            data[12].append(c.doc_type[:32] if c.doc_type else "3gpp")
            data[13].append(c.content_type[:32] if c.content_type else "")
            data[14].append(c.spec_role[:32] if c.spec_role else "")
            data[15].append(c.topic_domain[:32] if c.topic_domain else "")

        try:
            self._collection.insert(data)
            self._collection.flush()
            inserted = len(chunks)
            logger.debug("插入 %d 条记录", inserted)
            return inserted
        except MilvusException as e:
            logger.error("插入失败: %s", e)
            raise

    def delete_by_filter(self, filter_expr: str) -> int:
        """按过滤条件删除."""
        self._ensure_connected()
        if self._collection is None:
            return 0
        try:
            result = self._collection.delete(filter_expr)
            return result.delete_count if hasattr(result, "delete_count") else 0
        except MilvusException as e:
            logger.error("删除失败: %s", e)
            return 0

    # ── BM25 索引管理 ──

    def build_bm25(
        self,
        texts: list[str],
        doc_ids: list[str],
        spec_numbers: list[str],
        chunk_indices: list[int],
        metadata: list[dict] | None = None,
    ) -> None:
        """从完整语料构建 BM25 索引。"""
        self._bm25.build(
            texts, doc_ids, spec_numbers, chunk_indices, metadata=metadata,
        )
        self._bm25.save()

    def rebuild_bm25_from_collection(self) -> int:
        """从当前 Milvus collection 读取全部 texts 重建 BM25 索引.

        用于增量摄入后刷新混合检索能力.

        Returns:
            重建的文档数.
        """
        self._ensure_connected()
        if self._collection is None:
            return 0

        texts: list[str] = []
        doc_ids: list[str] = []
        spec_numbers: list[str] = []
        chunk_indices: list[int] = []
        metadata: list[dict] = []
        batch_size = 8000
        last_id = -1

        while True:
            try:
                results = self._collection.query(
                    expr=f"id > {last_id}",
                    output_fields=[
                        "id", "text", "doc_id", "spec_number", "chunk_index",
                        "release", "series", "doc_type",
                    ],
                    limit=batch_size,
                )
            except MilvusException as e:
                logger.error("BM25 重建查询失败: %s", e)
                if not texts:
                    return 0
                break

            if not results:
                break

            texts.extend(str(r.get("text", "")) for r in results)
            doc_ids.extend(str(r.get("doc_id", "")) for r in results)
            spec_numbers.extend(str(r.get("spec_number", "")) for r in results)
            chunk_indices.extend(int(r.get("chunk_index", 0)) for r in results)
            metadata.extend({
                "release": r.get("release", ""),
                "series": r.get("series", 0),
                "doc_type": r.get("doc_type", "3gpp"),
            } for r in results)
            last_id = results[-1]["id"]

            if len(results) < batch_size:
                break

        if not texts:
            return 0

        self.build_bm25(
            texts, doc_ids, spec_numbers, chunk_indices, metadata=metadata,
        )
        logger.info("BM25 索引已从 collection 重建 (%d 条)", len(texts))
        return len(texts)

    def load_bm25(self) -> bool:
        """从磁盘加载 BM25 索引。"""
        return self._bm25.load()

    @property
    def bm25_count(self) -> int:
        return self._bm25.doc_count

    # ── 检索 ──

    def search_dense(
        self, query_embedding: np.ndarray, top_k: int = 100,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Dense 向量相似度检索.

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter_expr: Milvus 标量过滤表达式, 如 'spec_number == "38.211"'
        """
        self._ensure_connected()
        if self._collection is None:
            return []

        query_vec = query_embedding.astype(np.float32).reshape(1, -1)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
        kwargs = {
            "data": query_vec.tolist(),
            "anns_field": "dense_vector",
            "param": search_params,
            "limit": top_k,
            "output_fields": [
                "text", "doc_id", "series", "spec_number",
                "release", "parent_section_id", "parent_title", "chunk_index",
                "section_number", "section_title", "section_path",
                "doc_type",
                "content_type", "spec_role", "topic_domain",
            ],
        }
        if filter_expr:
            kwargs["expr"] = filter_expr
        results = self._collection.search(**kwargs)

        output: list[SearchResult] = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                output.append(SearchResult(
                    chunk_id=hit.id,
                    text=entity.get("text", ""),
                    score=float(hit.distance),
                    doc_id=entity.get("doc_id", ""),
                    series=entity.get("series", 0),
                    spec_number=entity.get("spec_number", ""),
                    release=entity.get("release", ""),
                    parent_section_id=entity.get("parent_section_id", ""),
                    parent_title=entity.get("parent_title", ""),
                    chunk_index=entity.get("chunk_index", 0),
                    section_number=entity.get("section_number", ""),
                    section_title=entity.get("section_title", ""),
                    section_path=entity.get("section_path", ""),
                    doc_type=entity.get("doc_type", "3gpp"),
                    content_type=entity.get("content_type", ""),
                    spec_role=entity.get("spec_role", ""),
                    topic_domain=entity.get("topic_domain", ""),
                ))
        return output

    def search_sparse(
        self, query_text: str, top_k: int = 100,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """BM25 稀疏检索 (Python rank-bm25), 支持 Python 侧标量过滤."""
        self._ensure_connected()
        if not self._bm25.is_loaded:
            logger.warning("BM25 索引未加载")
            return []

        bm25_results = self._bm25.search_with_meta(query_text, top_k)

        output: list[SearchResult] = []
        for doc_key, score, text, meta in bm25_results:
            if not _matches_filter_expr(meta, filter_expr):
                continue
            # 解析 doc_key: "doc_id|spec_number|chunk_index"
            parts = doc_key.split("|", 2)
            doc_id = parts[0] if len(parts) > 0 else ""
            spec_number = parts[1] if len(parts) > 1 else ""
            chunk_index = int(parts[2]) if len(parts) > 2 else 0
            output.append(SearchResult(
                chunk_id=doc_key,
                text=text,
                score=score,
                doc_id=doc_id,
                spec_number=spec_number,
                chunk_index=chunk_index,
                series=meta.get("series", 0),
                release=meta.get("release", ""),
                doc_type=meta.get("doc_type", "3gpp"),
            ))
        return output

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        dense_top_k: int = 100,
        sparse_top_k: int = 100,
        final_top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """Dense + BM25 混合检索，Python 侧 RRF 融合。

        注意: BM25 (Python rank-bm25) 不支持 Milvus 标量过滤。
        当有 filter_expr 时跳过 BM25，仅 Dense 检索以保证过滤正确性。
        """
        self._ensure_connected()
        if self._collection is None:
            return []

        # 1. Dense 检索 (Milvus, 支持标量过滤)
        dense_results = self.search_dense(query_embedding, dense_top_k, filter_expr=filter_expr)

        # 2. BM25 检索 (Python, 按元数据做同样的标量过滤)
        sparse_results = self.search_sparse(query_text, sparse_top_k, filter_expr=filter_expr)

        if not sparse_results:
            # BM25 不可用时降级为纯 Dense
            return dense_results[:final_top_k]

        # 3. RRF 融合
        fused = self._rrf_fuse(
            dense_results, sparse_results, final_top_k,
            k_dense=settings.rrf_k_dense,
            k_sparse=settings.rrf_k_sparse,
        )
        return fused

    @staticmethod
    def _make_key(result: SearchResult) -> str:
        """生成文档唯一键: doc_id|spec_number|chunk_index."""
        return f"{result.doc_id}|{result.spec_number}|{result.chunk_index}"

    @staticmethod
    def _rrf_fuse(
        dense: list[SearchResult],
        sparse: list[SearchResult],
        final_top_k: int,
        k_dense: int = 60,
        k_sparse: int = 60,
    ) -> list[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 融合 Dense + BM25 结果.

        RRF(d) = sum_{r in rankings} 1 / (k_r + rank(d, r))

        k 值越小该路排名的贡献越大: 3GPP 领域 Dense 通常优于 BM25,
        可设 k_dense=40 / k_sparse=120 提高 Dense 权重 (需评测集回归验证).
        """
        # 建立 dense 排名表: doc_key -> rank (1-based)
        dense_rank: dict[str, int] = {}
        for i, r in enumerate(dense):
            key = MilvusStore._make_key(r)
            if key not in dense_rank:
                dense_rank[key] = i + 1

        # 建立 sparse 排名表
        sparse_rank: dict[str, int] = {}
        sparse_map: dict[str, SearchResult] = {}
        for i, r in enumerate(sparse):
            key = MilvusStore._make_key(r)
            if key not in sparse_rank:
                sparse_rank[key] = i + 1
            sparse_map[key] = r

        # 合并所有 keys
        all_keys = set(dense_rank.keys()) | set(sparse_rank.keys())

        # 计算 RRF 分数
        rrf_scores: dict[str, float] = {}
        for key in all_keys:
            rrf = 0.0
            if key in dense_rank:
                rrf += 1.0 / (k_dense + dense_rank[key])
            if key in sparse_rank:
                rrf += 1.0 / (k_sparse + sparse_rank[key])
            rrf_scores[key] = rrf

        # 按 RRF 分数排序
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:final_top_k]

        # 构建结果：优先使用 Dense 结果的完整元数据
        dense_map: dict[str, SearchResult] = {}
        for r in dense:
            key = MilvusStore._make_key(r)
            if key not in dense_map:
                dense_map[key] = r

        results: list[SearchResult] = []
        for key in sorted_keys:
            if key in dense_map:
                r = dense_map[key]
                results.append(SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=rrf_scores[key],
                    doc_id=r.doc_id,
                    series=r.series,
                    spec_number=r.spec_number,
                    release=r.release,
                    parent_section_id=r.parent_section_id,
                    parent_title=r.parent_title,
                    chunk_index=r.chunk_index,
                    doc_type=r.doc_type,
                    content_type=r.content_type,
                    spec_role=r.spec_role,
                    topic_domain=r.topic_domain,
                ))
            elif key in sparse_map:
                r = sparse_map[key]
                results.append(SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=rrf_scores[key],
                    doc_id=r.doc_id,
                    spec_number=r.spec_number,
                    chunk_index=r.chunk_index,
                    series=r.series,
                    doc_type=r.doc_type,
                    content_type=r.content_type,
                    spec_role=r.spec_role,
                    topic_domain=r.topic_domain,
                ))

        return results

    # ── 属性 ──

    @property
    def count(self) -> int:
        self._ensure_connected()
        if self._collection is None:
            return 0
        try:
            return self._collection.num_entities
        except MilvusException:
            return 0

    @property
    def supports_bm25(self) -> bool:
        return self._bm25.is_loaded

    # ── 文档管理 (适配 REST API) ──

    def get_documents_summary(self) -> dict[str, dict]:
        """查询 Milvus 获取文档摘要 Map (按 doc_id 聚合).

        Returns:
            {doc_id: {doc_id, spec_number, release, title, series, chunk_count}}
        """
        self._ensure_connected()
        if self._collection is None:
            return {}

        doc_map: dict[str, dict] = {}
        batch_size = 8000
        last_id = -1

        while True:
            try:
                results = self._collection.query(
                    expr=f"id > {last_id}",
                    output_fields=["id", "doc_id", "spec_number", "release", "series", "parent_title", "doc_type"],
                    limit=batch_size,
                )
            except MilvusException as e:
                logger.error("查询文档摘要失败: %s", e)
                return {} if not doc_map else doc_map

            if not results:
                break

            for r in results:
                doc_id = str(r.get("doc_id", ""))
                if not doc_id:
                    continue
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc_id": doc_id,
                        "spec_number": str(r.get("spec_number", "")),
                        "release": str(r.get("release", "")),
                        "title": str(r.get("parent_title", "")),
                        "series": int(r.get("series", 0)),
                        "doc_type": str(r.get("doc_type", "3gpp")),
                        "chunk_count": 0,
                    }
                doc_map[doc_id]["chunk_count"] += 1

            last_id = results[-1]["id"]
            if len(results) < batch_size:
                break

        return doc_map

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """查询指定文档的所有 chunks (排序).

        Args:
            doc_id: 文档标识符.

        Returns:
            按 chunk_index 排序的 chunk 列表.
        """
        self._ensure_connected()
        if self._collection is None:
            return []

        try:
            results = self._collection.query(
                expr=f"doc_id == '{_escape_milvus_expr(doc_id)}'",
                output_fields=["text", "spec_number", "release", "series",
                               "parent_section_id", "parent_title", "chunk_index"],
                limit=10000,
            )
        except MilvusException as e:
            logger.error("查询文档 chunks 失败 (%s): %s", doc_id, e)
            return []

        return sorted(results, key=lambda r: int(r.get("chunk_index", 0)))

    def get_adjacent_chunks(
        self, doc_id: str, chunk_index: int, window: int = 2,
    ) -> list[SearchResult]:
        """获取同一文档中指定 chunk 的相邻 chunks.

        Args:
            doc_id: 文档标识符.
            chunk_index: 目标 chunk 序号.
            window: 左右各取 window 个相邻 chunk.

        Returns:
            相邻 chunk 列表 (排除目标 chunk 自身), 按 chunk_index 排序.
        """
        self._ensure_connected()
        if self._collection is None:
            return []

        lo = chunk_index - window
        hi = chunk_index + window
        try:
            results = self._collection.query(
                expr=(
                    f"doc_id == '{_escape_milvus_expr(doc_id)}'"
                    f" and chunk_index >= {lo}"
                    f" and chunk_index <= {hi}"
                    f" and chunk_index != {chunk_index}"
                ),
                output_fields=["text", "spec_number", "release", "series",
                               "parent_section_id", "parent_title", "chunk_index"],
                limit=window * 2 + 2,
            )
        except MilvusException as e:
            logger.warning("查询相邻 chunk 失败 (doc=%s idx=%d): %s", doc_id, chunk_index, e)
            return []

        # 转换为 SearchResult 并按 chunk_index 排序
        adjacent: list[SearchResult] = []
        for row in results:
            adjacent.append(SearchResult(
                chunk_id=row.get("id", row.get("chunk_index", 0)),
                text=row.get("text", ""),
                score=0.0,
                doc_id=row.get("doc_id", doc_id),
                series=row.get("series", 0),
                spec_number=row.get("spec_number", ""),
                release=row.get("release", ""),
                parent_section_id=row.get("parent_section_id", ""),
                parent_title=row.get("parent_title", ""),
                chunk_index=row.get("chunk_index", 0),
            ))
        adjacent.sort(key=lambda r: r.chunk_index)
        return adjacent
