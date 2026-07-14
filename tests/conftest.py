"""pytest 共享 fixtures — 为所有测试模块提供通用 mock 对象."""

import pytest

from src.retriever.search import RetrievalResult


@pytest.fixture
def make_chunk():
    """工厂函数: 创建标准 RetrievalResult 用于测试."""
    def _make(
        chunk_id="1",
        text="Default test chunk text for verification purposes.",
        spec_number="38.413",
        section="8.3.1",
        score=0.85,
        doc_id="doc1",
        series=38,
        release="R18",
        title="PDU Session",
        idx=0,
    ):
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            score=score,
            doc_id=doc_id,
            series=series,
            spec_number=spec_number,
            release=release,
            parent_section_id=section,
            parent_title=title,
            chunk_index=idx,
        )
    return _make


@pytest.fixture
def sample_chunks(make_chunk):
    """3 个典型的 3GPP 检索结果."""
    return [
        make_chunk("1", "PDU Session Resource Setup procedure NGAP protocol N2 interface", "38.413", "8.3.1"),
        make_chunk("2", "QoS Flow binding to DRB at SDAP layer", "38.413", "5.3.2", score=0.78),
        make_chunk("3", "PDU Session establishment in 5GS architecture", "23.501", "5.6.7", score=0.72),
    ]
