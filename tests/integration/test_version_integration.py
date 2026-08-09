"""version 字段集成测试 (BT) — mock Milvus 验证 version 写入与检索返回.

覆盖 release 感知问答升级后的版本粒度过滤链路:
  1. _insert_batch 将 chunk.version 写入 Milvus 数据列
  2. search_dense 输出字段含 version 且回填 SearchResult
  3. RetrievalResult 携带 version (供多版本对比)
  4. version 过滤表达式在 Dense (expr 透传) 与 BM25 (Python 侧) 双路生效
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retriever.milvus_store import MilvusStore, _matches_filter_expr
from src.retriever.search import RetrievalResult
from src.retriever.vector_store import Chunk

# _insert_batch 数据列顺序 (与 schema 字段顺序一致, 不含 auto id)
_INSERT_COLUMNS = [
    "text", "dense_vector", "doc_id", "series", "spec_number",
    "release", "version", "parent_section_id", "parent_title",
    "chunk_index", "section_number", "section_title", "section_path",
    "doc_type", "content_type", "spec_role", "topic_domain",
    "parent_chunk_id", "parent_text",
]


class _FakeHit:
    """模拟 pymilvus search 命中的 hit (id + distance + entity.get)."""

    def __init__(self, row: dict, score: float):
        self.id = 1
        self.distance = score
        self._row = row

    @property
    def entity(self) -> "_FakeEntity":
        return _FakeEntity(self._row)


class _FakeEntity:
    def __init__(self, row: dict):
        self._row = row

    def get(self, key: str, default=""):
        return self._row.get(key, default)


class _FakeCollection:
    """内存版 Milvus 集合 — 记录 insert 数据, 从已插入行生成 search hits."""

    def __init__(self):
        self._rows: list[dict] = []
        self.search_expr: str | None = None

    def insert(self, data: list[list]):
        for i in range(len(data[0])):
            self._rows.append({col: data[c][i] for c, col in enumerate(_INSERT_COLUMNS)})

    def flush(self) -> None:
        pass

    def search(self, data, anns_field, param, limit, output_fields, **kwargs):
        self.search_expr = kwargs.get("expr")
        hits = []
        for row in self._rows[:limit]:
            hits.append(_FakeHit(row, score=0.9))
        return [hits]


@pytest.fixture
def version_store() -> MilvusStore:
    """带内存 fake collection 的 MilvusStore (跳过真实连接)."""
    store = MilvusStore.__new__(MilvusStore)
    store._collection = _FakeCollection()
    store._bm25 = MagicMock()
    store._connected = True
    return store


class TestVersionWriteAndRetrieve:

    def test_version_written_to_insert_columns(self, version_store):
        """version 应写入 release 之后的列 (数据列顺序与 schema 一致)."""
        version_store.insert([
            Chunk(text="PRACH preamble formats", doc_id="d1", series=38,
                  spec_number="38.211", release="R18", version="18.4.0"),
            Chunk(text="Another table chunk", doc_id="d2", series=38,
                  spec_number="38.211", release="R18", version="18.3.0"),
        ])
        rows = version_store._collection._rows
        assert rows[0]["version"] == "18.4.0"
        assert rows[1]["version"] == "18.3.0"

    def test_version_missing_defaults_empty(self, version_store):
        """未提供 version 的 chunk 写入空字符串 (不阻塞摄入)."""
        version_store.insert([Chunk(text="plain", doc_id="d3", series=23, spec_number="23.501")])
        assert version_store._collection._rows[0]["version"] == ""

    def test_search_returns_version(self, version_store):
        """search_dense 应返回 version 字段并回填 SearchResult."""
        version_store.insert([Chunk(text="preamble formats", doc_id="d1", series=38,
                                    spec_number="38.211", release="R18", version="18.4.0")])
        results = version_store.search_dense(np.zeros(1024, dtype=np.float32), top_k=1)
        assert len(results) == 1
        assert results[0].version == "18.4.0"

    def test_retrieval_result_carries_version(self, version_store):
        """SearchResult → RetrievalResult 转换保留 version (多版本对比数据通路)."""
        version_store.insert([Chunk(text="preamble formats", doc_id="d1", series=38,
                                    spec_number="38.211", release="R18", version="18.4.0")])
        results = version_store.search_dense(np.zeros(1024, dtype=np.float32), top_k=1)
        r = RetrievalResult.from_search_result(results[0])
        assert r.version == "18.4.0"


class TestVersionFilter:

    def test_dense_filter_expr_passes_through(self, version_store):
        """Dense 侧 version 过滤表达式应透传给 Milvus (expr 参数)."""
        version_store.insert([Chunk(text="t", doc_id="d", version="18.4.0")])
        expr = 'release == "R18" && version == "18.4.0"'
        version_store.search_dense(np.zeros(1024, dtype=np.float32), top_k=5, filter_expr=expr)
        assert version_store._collection.search_expr == expr

    def test_bm25_python_side_version_filter(self):
        """BM25 侧 (Python rank-bm25) 用元数据做 version 过滤."""
        meta = {"release": "R18", "version": "18.4.0"}
        assert _matches_filter_expr(meta, 'version == "18.4.0"') is True
        assert _matches_filter_expr(meta, 'version == "18.3.0"') is False

    def test_version_discriminates_multi_version_corpus(self, version_store):
        """同一规范不同版本 chunk 并存时, version 可区分 (对比问答基础)."""
        for ver in ("18.3.0", "18.4.0"):
            version_store.insert([Chunk(text=f"content {ver}", doc_id=f"d-{ver}",
                                        series=38, spec_number="38.211",
                                        release="R18", version=ver)])
        rows = version_store._collection._rows
        versions = {r["version"] for r in rows}
        assert versions == {"18.3.0", "18.4.0"}
