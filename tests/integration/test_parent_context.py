"""small-to-big 集成测试 — mock Milvus 验证 insert 含 parent 字段.

摄入侧: _insert_batch 将 parent_chunk_id/parent_text 写入 Milvus (18 列 schema).
检索侧: search_dense 输出字段含 parent_text, 经 RetrievalResult 透传.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retriever.milvus_store import MilvusStore
from src.retriever.vector_store import Chunk, SearchResult


@pytest.fixture
def mock_store() -> MagicMock:
    store = MagicMock()
    store._collection = MagicMock()
    return store


class TestMilvusInsertParentFields:
    def test_insert_batch_carries_parent_fields(self, mock_store):
        store = MilvusStore.__new__(MilvusStore)
        store._collection = mock_store._collection
        store._insert_batch([
            Chunk(
                text="sub chunk", doc_id="d", spec_number="38.413",
                parent_section_id="5.3.2", parent_title="RRC Setup",
                parent_chunk_id=12, parent_text="Section full context...",
            ),
        ])
        data = mock_store._collection.insert.call_args[0][0]
        assert len(data) == 18
        assert data[16] == [12]
        assert data[17] == ["Section full context..."]

    def test_search_dense_returns_parent_fields(self, mock_store):
        hit = MagicMock()
        hit.id = 5
        hit.distance = 0.9
        hit.entity.get.side_effect = lambda k, d=None: {
            "text": "sub chunk", "doc_id": "d", "series": 38,
            "spec_number": "38.413", "release": "R18",
            "parent_section_id": "5.3.2", "parent_title": "RRC Setup",
            "chunk_index": 1, "section_number": "", "section_title": "",
            "section_path": "", "doc_type": "3gpp", "content_type": "",
            "spec_role": "", "topic_domain": "",
            "parent_chunk_id": 12, "parent_text": "Section full context...",
        }.get(k, d)

        collection = MagicMock()
        collection.search.return_value = [[hit]]
        store = MilvusStore.__new__(MilvusStore)
        store._collection = collection
        store._connected = True

        out = store.search_dense(np.zeros(1024, dtype=np.float32), top_k=5)
        assert len(out) == 1
        assert out[0].parent_text == "Section full context..."
        assert out[0].parent_chunk_id == 12

    def test_hybrid_rrf_preserves_parent_fields(self, mock_store):
        from src.retriever.milvus_store import MilvusStore as MS
        dense = [
            SearchResult(
                chunk_id=1, text="t", score=0.9, doc_id="d",
                spec_number="38.413", chunk_index=1,
                parent_chunk_id=12, parent_text="parent ctx",
            ),
        ]
        fused = MS._rrf_fuse(dense, [], 5)
        assert fused[0].parent_text == "parent ctx"
        assert fused[0].parent_chunk_id == 12


class TestSmallToBigRetrieval:
    """检索侧 small-to-big — 子 chunk 命中 → 父 section 文本进入 LLM 上下文."""

    def test_parent_text_flows_to_prompt_context(self):
        from src.generator.prompt import build_rag_prompt
        from src.retriever.planner import RetrievalPlanner
        from src.retriever.search import RetrievalResult

        planner = RetrievalPlanner.__new__(RetrievalPlanner)
        results = [RetrievalResult(
            chunk_id=1, text="sub chunk hit", score=0.9, doc_id="d",
            series=38, spec_number="38.413", release="R18",
            parent_section_id="5.3", parent_title="RRC Setup",
            parent_text="Parent section complete text for context",
        )]

        planner.expand_parent_context(results)
        assert results[0].parent_context == "Parent section complete text for context"

        prompt = build_rag_prompt("RRC setup query", results)
        user = prompt[1]["content"]
        assert "Parent section complete text for context" in user

    def test_search_result_preserves_parent_via_from_search_result(self):
        from src.retriever.search import RetrievalResult
        sr = SearchResult(
            chunk_id=9, text="sub", score=0.8, doc_id="d",
            spec_number="38.413", parent_section_id="5.3",
            parent_chunk_id=12, parent_text="parent full text",
        )
        rr = RetrievalResult.from_search_result(sr)
        assert rr.parent_text == "parent full text"
        assert rr.parent_chunk_id == 12
