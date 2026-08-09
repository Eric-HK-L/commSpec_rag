"""graph_expander.py 单元测试 — 图扩展 chunk 的补充分数 (rank-based) 与来源标记."""

import json
from unittest.mock import MagicMock

from src.retriever.graph_expander import GraphExpander
from src.retriever.search import RetrievalResult
from src.retriever.vector_store import SearchResult


def _write_graph(tmp_path, adjacency=None, nodes=None):
    graph_path = tmp_path / "xref_graph.json"
    graph_path.write_text(json.dumps({
        "metadata": {"total_nodes": len(nodes or {}), "total_edges": 0},
        "adjacency": adjacency or {},
        "nodes": nodes or [],
    }, ensure_ascii=False), encoding="utf-8")
    return graph_path


def _fake_store(chunk_row: dict):
    store = MagicMock()
    store._collection.query.return_value = [chunk_row]
    return store


def _hit(chunk_id=5, spec_number="38.300"):
    return RetrievalResult(
        chunk_id=chunk_id, text="hit", score=0.9,
        spec_number=spec_number, parent_section_id="5.1.2",
    )


def _chunk_row(chunk_id=101, spec_number="38.331"):
    return {
        "id": chunk_id, "text": "referenced chunk", "doc_id": "doc2",
        "series": 38, "spec_number": spec_number, "release": "R18",
        "parent_section_id": "6.3.1", "parent_title": "Ref Section",
        "chunk_index": 0, "section_number": "", "section_title": "",
        "section_path": "",
    }


class TestGraphExpanderScores:
    """图扩展 chunk 不再硬编码 score=0.0, 而是带真实可排序的 rank-based 分数."""

    def test_expanded_chunk_has_real_rank_score(self, tmp_path):
        graph_path = _write_graph(
            tmp_path,
            adjacency={"5": {"REFERENCES": ["101"]}},
            nodes=[{"id": "101", "spec": "38.331"}],
        )
        expander = GraphExpander(graph_path)
        store = _fake_store(_chunk_row(chunk_id=101, spec_number="38.331"))
        expanded = expander.expand([_hit()], max_per_chunk=2, top_n=1, store=store)
        assert len(expanded) == 1
        assert expanded[0].score > 0.0, "图扩展 chunk 分数为 0.0, 无法排序/评分"
        assert expanded[0]._source_tag == "graph_expand"

    def test_scores_decrease_with_discovery_rank(self, tmp_path):
        graph_path = _write_graph(
            tmp_path,
            adjacency={"5": {"REFERENCES": ["101", "102"]}},
            nodes=[
                {"id": "101", "spec": "38.331"},
                {"id": "102", "spec": "23.501"},
            ],
        )
        expander = GraphExpander(graph_path)
        store = MagicMock()
        store._collection.query.side_effect = [
            [_chunk_row(chunk_id=101, spec_number="38.331")],
            [_chunk_row(chunk_id=102, spec_number="23.501")],
        ]
        expanded = expander.expand([_hit()], max_per_chunk=2, top_n=1, store=store)
        assert len(expanded) == 2
        assert expanded[0].score > expanded[1].score, "rank-based 分数应随发现次序递减"

    def test_rank_score_is_rrf_comparable(self, tmp_path):
        graph_path = _write_graph(
            tmp_path,
            adjacency={"5": {"REFERENCES": ["101"]}},
            nodes=[{"id": "101", "spec": "38.331"}],
        )
        expander = GraphExpander(graph_path)
        store = _fake_store(_chunk_row(chunk_id=101, spec_number="38.331"))
        expanded = expander.expand([_hit()], max_per_chunk=1, top_n=1, store=store)
        # 与 multi_hop 的 RRF 尺度 (1/(60+rank), max≈0.033) 同量级
        assert 0.01 <= expanded[0].score <= 0.05


class TestSourceTagPreservation:
    """RetrievalResult.from_search_result 保留动态 _source_tag, 供下游识别补充通道."""

    def test_from_search_result_preserves_source_tag(self):
        sr = SearchResult(chunk_id=101, text="x", score=0.016)
        sr._source_tag = "graph_expand"
        rr = RetrievalResult.from_search_result(sr)
        assert getattr(rr, "_source_tag", None) == "graph_expand", \
            "_source_tag 在 from_search_result 转换时丢失, 补充通道无法被下游识别"

    def test_from_search_result_untagged_stays_untagged(self):
        sr = SearchResult(chunk_id=1, text="x", score=0.9)
        rr = RetrievalResult.from_search_result(sr)
        assert getattr(rr, "_source_tag", None) is None
