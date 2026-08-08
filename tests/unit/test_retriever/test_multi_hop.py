"""multi_hop.py 单元测试 — 纯函数逻辑."""

from src.retriever.multi_hop import (
    _build_chunk_summary,
    _build_gap_analysis_prompt,
    _merge_results,
    _parse_gap_response,
    needs_multi_hop,
)
from src.retriever.search import RetrievalResult


def _make_chunk(chunk_id="1", spec_number="38.413", section="8.3.1", text="Test chunk", score=0.8):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=score,
        doc_id="doc1", series=38, spec_number=spec_number,
        release="R18", parent_section_id=section,
        parent_title="Test", chunk_index=0,
    )


class TestParseGapResponse:

    def test_sufficient(self):
        is_suf, queries = _parse_gap_response("SUFFICIENT")
        assert is_suf is True
        assert queries == []

    def test_sufficient_lowercase(self):
        is_suf, queries = _parse_gap_response("sufficient")
        assert is_suf is True

    def test_sufficient_with_extra(self):
        is_suf, queries = _parse_gap_response("SUFFICIENT\nNo more info needed.")
        assert is_suf is True
        assert queries == []  # 忽略 SUFFICIENT 后的内容

    def test_sub_queries(self):
        response = """NR PUSCH 的 DMRS 配置方式
NR 载波聚合中 PUCCH group 的定义"""
        is_suf, queries = _parse_gap_response(response)
        assert is_suf is False
        assert len(queries) == 2
        assert "DMRS" in queries[0]

    def test_sub_queries_max_limit(self):
        # 超过 MAX_SUB_QUERIES (4) 应截断
        lines = [f"Sub query line {i}" for i in range(10)]
        response = "\n".join(lines)
        is_suf, queries = _parse_gap_response(response)
        assert len(queries) <= 4

    def test_filter_short_lines(self):
        response = "OK\nA valid sub query with enough text\nAB"
        is_suf, queries = _parse_gap_response(response)
        assert all(len(q) > 5 for q in queries)


class TestBuildChunkSummary:

    def test_basic_summary(self):
        chunks = [
            _make_chunk("1", "38.413", "8.3.1", "PDU Session Resource Setup procedure"),
            _make_chunk("2", "23.501", "5.6.7", "PDU Session establishment"),
        ]
        summary = _build_chunk_summary(chunks)
        assert "[38.413 §8.3.1]" in summary
        assert "[23.501 §5.6.7]" in summary
        assert "PDU Session" in summary

    def test_max_chunks_truncation(self):
        chunks = [_make_chunk(str(i), f"38.{i}00", text=f"chunk {i}") for i in range(20)]
        summary = _build_chunk_summary(chunks, max_chunks=5)
        lines = summary.split("\n")
        assert len(lines) <= 5

    def test_empty_chunks(self):
        summary = _build_chunk_summary([])
        assert summary == ""

    def test_missing_spec_number(self):
        chunk = _make_chunk(spec_number="")
        summary = _build_chunk_summary([chunk])
        assert "[? §" in summary  # fallback


class TestBuildGapAnalysisPrompt:

    def test_structure(self):
        msgs = _build_gap_analysis_prompt("How does PDU Session work?", "summary text")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "3GPP 规范检索专家" in msgs[0]["content"]
        assert "How does PDU Session work?" in msgs[1]["content"]
        assert "summary text" in msgs[1]["content"]


class TestMergeResults:

    def test_basic_merge(self):
        original = [_make_chunk("1"), _make_chunk("2")]
        supplement = [_make_chunk("3"), _make_chunk("4")]
        merged = _merge_results(original, supplement)
        assert len(merged) == 4
        assert merged[0].chunk_id == "1"
        assert merged[2].chunk_id == "3"

    def test_deduplication(self):
        original = [_make_chunk("1"), _make_chunk("2")]
        supplement = [_make_chunk("2"), _make_chunk("3")]  # "2" 重复
        merged = _merge_results(original, supplement)
        assert len(merged) == 3
        ids = {r.chunk_id for r in merged}
        assert ids == {"1", "2", "3"}

    def test_source_tag(self):
        original = [_make_chunk("1")]
        supplement = [_make_chunk("2")]
        merged = _merge_results(original, supplement)
        assert getattr(merged[1], "_source_tag", None) == "multi_hop"

    def test_original_order_preserved(self):
        original = [_make_chunk("2"), _make_chunk("1")]
        supplement = [_make_chunk("3")]
        merged = _merge_results(original, supplement)
        assert merged[0].chunk_id == "2"
        assert merged[1].chunk_id == "1"


class TestNeedsMultiHop:

    def test_low_diversity(self):
        chunks = [_make_chunk(spec_number="38.413") for _ in range(4)]
        assert needs_multi_hop(chunks) is True  # diversity = 1/4 = 0.25

    def test_high_diversity(self):
        chunks = [
            _make_chunk(spec_number="38.413"),
            _make_chunk(spec_number="23.501"),
            _make_chunk(spec_number="36.211"),
            _make_chunk(spec_number="22.011"),
        ]
        assert needs_multi_hop(chunks) is False  # diversity = 4/4 = 1.0

    def test_few_results(self):
        chunks = [_make_chunk("1")]
        assert needs_multi_hop(chunks) is False

    def test_empty(self):
        assert needs_multi_hop([]) is False
