"""search.py 单元测试 — RetrievalResult/Chunk dataclass 纯函数."""

from src.retriever.search import RetrievalResult
from src.retriever.vector_store import Chunk, SearchResult


class TestRetrievalResult:
    """RetrievalResult — 检索结果的格式化和构造."""

    def test_from_search_result_basic(self):
        sr = SearchResult(
            chunk_id=1, text="test text", score=0.95,
            doc_id="doc1", series=38, spec_number="38.413",
            release="R18", parent_section_id="8.1", parent_title="N2 Interface",
            chunk_index=0,
        )
        r = RetrievalResult.from_search_result(sr)
        assert r.chunk_id == 1
        assert r.text == "test text"
        assert r.score == 0.95
        assert r.spec_number == "38.413"
        assert r.series == 38

    def test_from_search_result_defaults(self):
        sr = SearchResult(chunk_id="abc", text="content", score=0.5)
        r = RetrievalResult.from_search_result(sr)
        assert r.doc_id == ""
        assert r.series == 0
        assert r.spec_number == ""

    def test_to_context_str_full(self):
        r = RetrievalResult(
            chunk_id=1, text="NGAP protocol details",
            score=0.95, spec_number="38.413",
            parent_section_id="8.1", parent_title="N2 Interface",
            release="R18",
        )
        ctx = r.to_context_str()
        assert "TS 38.413" in ctx
        assert "§8.1" in ctx
        assert "N2 Interface" in ctx
        assert "(R18)" in ctx
        assert "NGAP protocol details" in ctx

    def test_to_context_str_minimal(self):
        r = RetrievalResult(
            chunk_id=1, text="simple text", score=0.5,
            doc_id="doc1", spec_number="",
        )
        ctx = r.to_context_str()
        assert "simple text" in ctx

    def test_to_context_str_index_param(self):
        r = RetrievalResult(
            chunk_id=1, text="text", score=0.5,
            spec_number="38.300",
        )
        # index 参数不影响输出格式 (当前实现未使用)
        ctx = r.to_context_str(index=3)
        assert "TS 38.300" in ctx

    def test_adjacent_chunks_default(self):
        r = RetrievalResult(chunk_id=1, text="text", score=0.5)
        assert r.adjacent_chunks == []


class TestChunkDataclass:
    """Chunk — 待入库文档片段."""

    def test_basic(self):
        c = Chunk(text="sample chunk text")
        assert c.text == "sample chunk text"
        assert c.embedding is None
        assert c.doc_id == ""
        assert c.chunk_index == 0

    def test_full_meta(self):
        c = Chunk(
            text="section content",
            doc_id="38.413-v1",
            series=38,
            spec_number="38413",
            release="R18",
            parent_section_id="8.1.2",
            parent_title="NG Setup",
            chunk_index=3,
        )
        assert c.series == 38
        assert c.spec_number == "38413"
        assert c.parent_title == "NG Setup"

    def test_version_default_empty(self):
        c = Chunk(text="t", doc_id="d")
        assert c.version == ""

    def test_version_set(self):
        c = Chunk(text="t", doc_id="d", version="18.4.0")
        assert c.version == "18.4.0"


class TestVersionInRetrievalResult:
    """RetrievalResult 版本字段 — 构造默认 + from_search_result 传递."""

    def test_default_empty(self):
        r = RetrievalResult(chunk_id=1, text="t", score=0.5)
        assert r.version == ""

    def test_from_search_result_carries_version(self):
        sr = SearchResult(
            chunk_id=1, text="t", score=0.9, doc_id="d",
            spec_number="38.211", release="R18", version="18.4.0",
        )
        r = RetrievalResult.from_search_result(sr)
        assert r.version == "18.4.0"

    def test_from_search_result_missing_version_defaults(self):
        sr = SearchResult(chunk_id=1, text="t", score=0.9)
        r = RetrievalResult.from_search_result(sr)
        assert r.version == ""
