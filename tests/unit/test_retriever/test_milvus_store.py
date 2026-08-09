"""milvus_store.py 字段截断工具单元测试.

覆盖 VARCHAR 字段写入 Milvus 前的字节安全截断 (修复 marked 数据集 38.331
深嵌套章节 section_path 2411 字节 > schema 1024 上限导致的入库失败):
  - 短文本原样返回
  - 中文多字节文本按字节截断且不超上限
  - 截断处不产生 UTF-8 解码错误
"""

from unittest.mock import MagicMock

from src.retriever.milvus_store import (
    DataType,
    MilvusStore,
    _matches_filter_expr,
    _safe_truncate_bytes,
)
from src.retriever.vector_store import Chunk, SearchResult


class TestSafeTruncateBytes:

    def test_short_text_unchanged(self):
        s = "6.3.1  RRCReconfiguration"
        assert _safe_truncate_bytes(s, 1024) == s

    def test_long_chinese_path_truncated_to_limit(self):
        # 模拟 38.331 深嵌套中文章节路径: 800 字符 × 3 字节 = 2400 字节
        s = "无线资源控制信令流程 > RRC 重配置 > 信令承载配置 > 数据传输承载配置 " * 40
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024
        # 截断后有标记且 UTF-8 完整
        assert out.endswith("…")

    def test_ascii_path_truncated(self):
        s = "a" * 2000
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024

    def test_4096_limit_covers_deepest_path(self):
        # 当前数据集最深 section_path 2411 字节, 4096 上限必须完整容纳
        s = "章节" * 500  # 3000 字节
        out = _safe_truncate_bytes(s, 4096)
        assert len(out.encode("utf-8")) <= 4096


def _sr(doc_id: str, spec: str, idx: int, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc_id}|{spec}|{idx}", text="ctx", score=score,
        doc_id=doc_id, spec_number=spec, chunk_index=idx,
    )


class TestMatchesFilterExpr:
    """BM25 Python 侧标量过滤 — 覆盖 _build_filter_expr 生成的全部形态."""

    META = {"release": "R18", "series": 38, "doc_type": "3gpp"}

    def test_empty_expr_passes(self):
        assert _matches_filter_expr(self.META, "") is True
        assert _matches_filter_expr(self.META, None) is True

    def test_single_field_match(self):
        assert _matches_filter_expr(self.META, 'release == "R18"') is True
        assert _matches_filter_expr(self.META, 'series == 38') is True
        assert _matches_filter_expr(self.META, 'doc_type == "3gpp"') is True

    def test_single_field_mismatch(self):
        assert _matches_filter_expr(self.META, 'release == "R17"') is False
        assert _matches_filter_expr(self.META, 'series == 23') is False
        assert _matches_filter_expr(self.META, 'doc_type == "oran"') is False

    def test_and_combination(self):
        expr = 'release == "R18" && series == 38 && doc_type == "3gpp"'
        assert _matches_filter_expr(self.META, expr) is True
        assert _matches_filter_expr({"release": "R18", "series": 23, "doc_type": "3gpp"}, expr) is False

    def test_or_with_oran_fallback(self):
        expr = '((release == "R18" && series == 38 && doc_type == "3gpp") || doc_type == "oran")'
        assert _matches_filter_expr(self.META, expr) is True
        assert _matches_filter_expr({"release": "R17", "series": 38, "doc_type": "3gpp"}, expr) is False
        assert _matches_filter_expr({"doc_type": "oran"}, expr) is True

    def test_empty_meta_rejected(self):
        assert _matches_filter_expr({}, 'release == "R18"') is False


class TestRrfFuse:
    """RRF 融合 — k 参数化后的加权行为."""

    def test_equal_k_favors_both_rankings(self):
        dense = [_sr("d1", "38.413", 0, 0.9)]
        sparse = [_sr("d1", "38.413", 0, 0.8), _sr("d2", "23.501", 0, 0.7)]
        fused = MilvusStore._rrf_fuse(dense, sparse, 2)
        assert fused[0].chunk_id == "d1|38.413|0"  # 双路命中排第一

    def test_smaller_k_gives_higher_weight(self):
        # k_dense 远小于 k_sparse → Dense 单路 rank1 胜过 Sparse 单路 rank1
        dense_only = _sr("d1", "38.413", 0)
        sparse_only = _sr("d2", "23.501", 0)
        fused = MilvusStore._rrf_fuse(
            [dense_only], [sparse_only], 2,
            k_dense=10, k_sparse=1000,
        )
        assert fused[0].chunk_id == dense_only.chunk_id

    def test_larger_k_gives_lower_weight(self):
        dense_only = _sr("d1", "38.413", 0)
        sparse_only = _sr("d2", "23.501", 0)
        fused = MilvusStore._rrf_fuse(
            [dense_only], [sparse_only], 2,
            k_dense=1000, k_sparse=10,
        )
        assert fused[0].chunk_id == sparse_only.chunk_id

    def test_default_k_preserves_original_behavior(self):
        dense = [_sr("d1", "38.413", 0)]
        sparse = [_sr("d2", "23.501", 0)]
        fused = MilvusStore._rrf_fuse(dense, sparse, 2)
        # 等权时双路单 rank1 得分相同, 顺序由输入决定
        assert len(fused) == 2


class TestInsertBatchParentFields:
    """_insert_batch 同步 small-to-big parent 字段 (含 version, 共 19 列)."""

    def test_writes_parent_fields_and_truncates(self):
        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        chunk = Chunk(
            text="sub chunk", doc_id="d", spec_number="38.413",
            parent_section_id="8.3", parent_title="Setup",
            parent_chunk_id=3,
            parent_text="父上下文" * 1500,  # 4500 bytes > VARCHAR 4096 上限
        )
        store._insert_batch([chunk])

        data = store._collection.insert.call_args[0][0]
        assert len(data) == 19, "schema 16 列 + parent_chunk_id + parent_text + version = 19 列"
        assert data[6] == [""], "version (VARCHAR) 应写入第 7 列, 默认空字符串"
        assert data[17] == [3], "parent_chunk_id (INT64) 应写入第 18 列"
        stored = data[18][0]
        assert len(stored.encode("utf-8")) <= 4096, "parent_text 必须 ≤ 4096 字节"
        assert stored.endswith("…"), "超限时应在语义边界截断并附加标记"

    def test_empty_parent_writes_defaults(self):
        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        store._insert_batch([Chunk(text="plain chunk", doc_id="d")])

        data = store._collection.insert.call_args[0][0]
        assert len(data) == 19
        assert data[6] == [""]
        assert data[17] == [0]
        assert data[18] == [""]


class TestVersionField:
    """Milvus 集合 version 字段 — schema 声明 + _insert_batch 写入."""

    def test_schema_declares_version_field(self, monkeypatch):
        """create_collection 的 schema 应含 version 字段 (VARCHAR 32, 位于 release 之后)."""
        import src.retriever.milvus_store as ms

        captured: dict = {}

        class _FakeCollection:
            def __init__(self, name, schema=None, **kwargs):
                if schema is not None:
                    captured["schema"] = schema

            def create_index(self, field_name, index_params):
                pass

            def load(self):
                pass

        monkeypatch.setattr(ms.connections, "connect", lambda **kw: None)
        monkeypatch.setattr(ms.utility, "has_collection", lambda name: False)
        monkeypatch.setattr(ms, "Collection", _FakeCollection)

        store = MilvusStore()
        store.create_collection(drop_existing=True)

        fields = captured["schema"].fields
        names = [f.name for f in fields]
        assert "version" in names
        version_field = next(f for f in fields if f.name == "version")
        assert version_field.dtype == DataType.VARCHAR
        assert version_field.max_length == 32
        # version 紧跟 release 之后 (字段顺序与 _insert_batch 列序一致)
        assert names.index("version") == names.index("release") + 1

    def test_insert_batch_writes_version_column(self):
        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        chunk = Chunk(text="table chunk", doc_id="d", release="R18", version="18.4.0")
        store._insert_batch([chunk])

        data = store._collection.insert.call_args[0][0]
        assert len(data) == 19
        assert data[6] == ["18.4.0"], "version 应写入 release 之后的列"

    def test_insert_batch_version_truncated_to_32(self):
        """version 超 32 字符时截断到 VARCHAR(32) 上限."""
        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        long_version = "v" * 40
        store._insert_batch([Chunk(text="t", doc_id="d", version=long_version)])

        data = store._collection.insert.call_args[0][0]
        assert len(data[6][0]) == 32

    def test_insert_batch_missing_version_defaults_empty(self):
        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        store._insert_batch([Chunk(text="t", doc_id="d")])

        data = store._collection.insert.call_args[0][0]
        assert data[6] == [""]

    def test_search_dense_output_includes_version(self):
        """search_dense 输出字段含 version 并回填 SearchResult.version."""

        store = MilvusStore.__new__(MilvusStore)
        store._collection = MagicMock()
        store._bm25 = MagicMock()
        store._connected = True

        def _fake_search(**kwargs):
            assert "version" in kwargs["output_fields"], "search 输出字段必须包含 version"
            hits = [
                MagicMock(
                    id=1,
                    distance=0.9,
                    entity=MagicMock(
                        get=lambda k, d="": {
                            "text": "t", "doc_id": "d", "series": 38,
                            "spec_number": "38.211", "release": "R18",
                            "version": "18.4.0",
                            "parent_section_id": "", "parent_title": "",
                            "chunk_index": 0, "section_number": "",
                            "section_title": "", "section_path": "",
                            "doc_type": "3gpp", "content_type": "",
                            "spec_role": "", "topic_domain": "",
                            "parent_chunk_id": 0, "parent_text": "",
                        }.get(k, d),
                    ),
                )
            ]
            return [hits]

        store._collection.search = _fake_search
        results = store.search_dense(__import__("numpy").zeros(1024, dtype="float32"), top_k=1)
        assert results[0].version == "18.4.0"
