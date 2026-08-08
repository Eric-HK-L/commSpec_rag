"""REST schemas.py 单元测试 — Pydantic 模型验证."""

import pytest
from pydantic import ValidationError

from src.api.rest.schemas import (
    APIResponse,
    ChunkItem,
    DocumentDetail,
    DocumentItem,
    PaginationParams,
    SearchFilters,
    SortParams,
    SystemStats,
)


class TestAPIResponse:
    """APIResponse — 统一响应包裹器."""

    def test_ok(self):
        resp = APIResponse.ok(data={"key": "value"}, took_ms=150)
        assert resp.success is True
        assert resp.data == {"key": "value"}
        assert resp.error is None
        assert resp.meta == {"took_ms": 150}

    def test_fail(self):
        resp = APIResponse.fail(error="Not Found", code=404)
        assert resp.success is False
        assert resp.data is None
        assert resp.error == "Not Found"
        assert resp.meta == {"code": 404}

    def test_default_meta(self):
        resp = APIResponse.ok(data="test")
        assert resp.meta == {}


class TestPaginationParams:
    """PaginationParams — 分页参数验证."""

    def test_defaults(self):
        p = PaginationParams()
        assert p.offset == 0
        assert p.limit == 20

    def test_custom(self):
        p = PaginationParams(offset=10, limit=50)
        assert p.offset == 10
        assert p.limit == 50

    def test_negative_offset(self):
        with pytest.raises(ValidationError):
            PaginationParams(offset=-1)

    def test_zero_limit(self):
        with pytest.raises(ValidationError):
            PaginationParams(limit=0)

    def test_limit_exceeds_max(self):
        with pytest.raises(ValidationError):
            PaginationParams(limit=201)


class TestSearchFilters:
    """SearchFilters — 检索过滤条件."""

    def test_empty(self):
        f = SearchFilters()
        assert f.series is None
        assert f.release is None
        assert f.spec_number is None

    def test_with_series(self):
        f = SearchFilters(series="38")
        assert f.series == "38"


class TestSortParams:
    """SortParams — 排序参数."""

    def test_defaults(self):
        s = SortParams()
        assert s.sort_by == "score"
        assert s.sort_order == "desc"

    def test_valid_asc(self):
        s = SortParams(sort_order="asc")
        assert s.sort_order == "asc"

    def test_invalid_order(self):
        with pytest.raises(ValidationError):
            SortParams(sort_order="random")


class TestDocumentModels:
    """Pydantic 数据模型验证."""

    def test_document_item(self):
        d = DocumentItem(doc_id="d1", spec_number="38.413", release="R18")
        assert d.doc_id == "d1"
        assert d.spec_number == "38.413"
        assert d.title == ""  # default

    def test_document_detail(self):
        d = DocumentDetail(
            doc_id="d1", spec_number="38.413", release="R18",
            version="v17.0.0", source="docx",
        )
        assert d.version == "v17.0.0"
        assert d.source == "docx"
        assert d.release == "R18"

    def test_chunk_item(self):
        c = ChunkItem(
            chunk_id=42, text="content", spec_number="38.413", release="R18",
        )
        assert c.chunk_id == 42
        assert c.text == "content"

    def test_system_stats(self):
        s = SystemStats(
            total_docs=100, total_chunks=5000,
            releases={"R18": 95, "R17": 5},
            series_distribution={"38": 3000, "23": 2000},
            vector_db="milvus",
        )
        assert s.total_docs == 100
        assert s.vector_db == "milvus"
        assert s.embedding_dim == 1024
