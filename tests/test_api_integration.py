"""REST API 集成测试 — FastAPI TestClient 全端点验证."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.rest.router import router, set_pipeline
from src.api.rest.schemas import (
    APIResponse,
    ChunkItem,
    DocumentItem,
    SearchFilters,
    SystemStats,
)
from src.generator.pipeline import RAGPipeline, RAGResponse
from src.retriever.search import RetrievalResult


# ── Fixtures ──


@pytest.fixture
def mock_pipeline():
    """创建 mock RAGPipeline."""
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline._store = MagicMock()
    pipeline._store.count = 1000
    pipeline._store.__class__.__name__ = "MilvusStore"

    # mock ask() 返回
    def _ask(query, reranker_enabled=True, **kwargs):
        return RAGResponse(
            query=query,
            answer="The PDU Session Resource Setup is defined in TS 38.413 §8.3.1.",
            sources=[
                RetrievalResult(
                    chunk_id="1", text="PDU Session Resource Setup NGAP",
                    score=0.92, doc_id="d1", series=38,
                    spec_number="38.413", release="R18",
                    parent_section_id="8.3.1", parent_title="PDU Session Setup",
                    chunk_index=0,
                ),
            ],
            verified=True,
            warnings=[],
            coverage=0.75,
            expanded_query="PDU session setup 38.413",
        )
    pipeline.ask = _ask

    # mock ask_stream() — 流式生成器: sources → chunks → done
    def _ask_stream(query, reranker_enabled=True, **kwargs):
        yield ("sources", [
            RetrievalResult(
                chunk_id="1", text="PDU Session Resource Setup NGAP",
                score=0.92, doc_id="d1", series=38,
                spec_number="38.413", release="R18",
                parent_section_id="8.3.1", parent_title="PDU Session Setup",
                chunk_index=0,
            ),
        ])
        yield ("chunk", "The PDU Session Resource Setup is defined")
        yield ("chunk", " in TS 38.413 §8.3.1.")
        yield ("done", {
            "answer": "The PDU Session Resource Setup is defined in TS 38.413 §8.3.1.",
            "verified": True, "warnings": [], "coverage": 0.75,
            "expanded_query": "PDU session setup 38.413",
        })
    pipeline.ask_stream = _ask_stream

    # mock search() 返回
    def _search(query, top_k=10, reranker_enabled=True, **kwargs):
        return [
            RetrievalResult(
                chunk_id="1", text="PDU Session Resource Setup NGAP",
                score=0.92, doc_id="d1", series=38,
                spec_number="38.413", release="R18",
                parent_section_id="8.3.1", parent_title="PDU Session Setup",
                chunk_index=0,
            ),
            RetrievalResult(
                chunk_id="2", text="QoS Flow binding DRB",
                score=0.78, doc_id="d1", series=38,
                spec_number="38.413", release="R18",
                parent_section_id="5.3.2", parent_title="QoS Flow",
                chunk_index=1,
            ),
        ]
    pipeline.search = _search

    # mock store 方法
    pipeline._store.get_documents_summary.return_value = {
        "d1": {"doc_id": "d1", "spec_number": "38.413", "release": "R18",
               "title": "NG-RAN NGAP", "series": 38, "chunk_count": 200},
        "d2": {"doc_id": "d2", "spec_number": "23.501", "release": "R18",
               "title": "5GS Architecture", "series": 23, "chunk_count": 150},
    }

    pipeline._store.get_document_chunks.return_value = [
        {"id": 1, "text": "chunk1", "spec_number": "38.413", "release": "R18",
         "series": 38, "parent_section_id": "8.1", "parent_title": "Setup", "chunk_index": 0},
    ]
    pipeline._store.delete_by_filter.return_value = 5

    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """创建 TestClient 并注入 mock pipeline."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    # 注入 mock pipeline
    set_pipeline(mock_pipeline)

    with TestClient(app) as c:
        yield c


# ── 健康检查 ──


class TestHealthEndpoint:
    """GET /api/v1/health"""

    def test_health_ready(self, client, mock_pipeline):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["vector_db"] == "MilvusStore"
        assert data["chunk_count"] == 1000

    def test_health_initializing(self):
        """pipeline 未注入时返回 initializing."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        # 不注入 pipeline — 模拟启动中状态
        import src.api.rest.router as r
        old = r._pipeline
        r._pipeline = None
        try:
            with TestClient(app) as c:
                resp = c.get("/api/v1/health")
                assert resp.status_code == 200
                assert resp.json()["status"] == "initializing"
        finally:
            r._pipeline = old


# ── Search 端点 ──


class TestSearchEndpoint:
    """POST /api/v1/search"""

    def test_search_basic(self, client):
        resp = client.post("/api/v1/search", json={
            "query": "PDU session setup",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["query"] == "PDU session setup"
        assert len(data["data"]["results"]) > 0
        assert data["data"]["total"] == len(data["data"]["results"])

    def test_search_with_filters(self, client):
        resp = client.post("/api/v1/search", json={
            "query": "PDU session",
            "top_k": 5,
            "filters": {"release": "R18", "series": "38"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_search_empty_query(self, client):
        resp = client.post("/api/v1/search", json={
            "query": "",
        })
        assert resp.status_code == 422  # validation error

    def test_search_query_too_long(self, client):
        resp = client.post("/api/v1/search", json={
            "query": "x" * 2001,
        })
        assert resp.status_code == 422

    def test_search_top_k_too_large(self, client):
        resp = client.post("/api/v1/search", json={
            "query": "test",
            "top_k": 101,
        })
        assert resp.status_code == 422

    def test_search_503_when_no_pipeline(self):
        """未注入 pipeline 时返回 503."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        import src.api.rest.router as r
        old = r._pipeline
        r._pipeline = None
        try:
            with TestClient(app) as c:
                resp = c.post("/api/v1/search", json={"query": "test"})
                assert resp.status_code == 503
        finally:
            r._pipeline = old

    def test_search_pipeline_error(self, client, mock_pipeline):
        """pipeline.search 抛异常时返回 500."""
        mock_pipeline.search = MagicMock(side_effect=RuntimeError("DB error"))
        resp = client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 500


# ── Ask 端点 ──


class TestAskEndpoint:
    """POST /api/v1/ask"""

    def test_ask_basic(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "What is PDU Session Resource Setup?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is PDU Session Resource Setup?"
        assert len(data["answer"]) > 0
        assert data["verified"] is True
        assert isinstance(data["warnings"], list)
        assert len(data["sources"]) > 0
        assert "coverage" in data
        assert "expanded_query" in data

    def test_ask_with_reranker_disabled(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "PDU session",
            "reranker_enabled": False,
        })
        assert resp.status_code == 200

    def test_ask_response_structure(self, client):
        resp = client.post("/api/v1/ask", json={"query": "PDU Session"})
        data = resp.json()
        # 验证 source 结构
        for src in data["sources"]:
            assert "chunk_id" in src
            assert "text" in src
            assert "score" in src
            assert "spec_number" in src
            assert "parent_section_id" in src
            assert "parent_title" in src

    def test_ask_empty_query(self, client):
        resp = client.post("/api/v1/ask", json={"query": ""})
        assert resp.status_code == 422

    def test_ask_query_too_long(self, client):
        resp = client.post("/api/v1/ask", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_ask_503_no_pipeline(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        import src.api.rest.router as r
        old = r._pipeline
        r._pipeline = None
        try:
            with TestClient(app) as c:
                resp = c.post("/api/v1/ask", json={"query": "test"})
                assert resp.status_code == 503
        finally:
            r._pipeline = old

    def test_ask_pipeline_error(self, client, mock_pipeline):
        mock_pipeline.ask = MagicMock(side_effect=RuntimeError("LLM error"))
        resp = client.post("/api/v1/ask", json={"query": "test"})
        assert resp.status_code == 500


# ── Search/Count 端点 ──


class TestSearchCountEndpoint:
    """POST /api/v1/search/count"""

    def test_count_basic(self, client):
        resp = client.post("/api/v1/search/count", json={
            "query": "PDU session",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["data"], int)
        assert data["data"] >= 0


# ── Batch Search 端点 ──


class TestSearchBatchEndpoint:
    """POST /api/v1/search/batch"""

    def test_batch_basic(self, client):
        resp = client.post("/api/v1/search/batch", json={
            "queries": [
                {"query": "PDU session"},
                {"query": "QoS flow"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["data"]) == 2
        for item in data["data"]:
            assert "query" in item
            assert "total" in item
            assert "results" in item

    def test_batch_too_many_queries(self, client):
        resp = client.post("/api/v1/search/batch", json={
            "queries": [{"query": f"q{i}"} for i in range(11)],
        })
        assert resp.status_code == 422

    def test_batch_empty(self, client):
        resp = client.post("/api/v1/search/batch", json={
            "queries": [],
        })
        assert resp.status_code == 422


# ── Documents 端点 ──


class TestDocumentsEndpoint:
    """GET /api/v1/documents"""

    def test_list_documents(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1

    def test_list_with_pagination(self, client):
        resp = client.get("/api/v1/documents?offset=0&limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) <= 1

    def test_list_with_series_filter(self, client):
        resp = client.get("/api/v1/documents?series=38")
        assert resp.status_code == 200
        data = resp.json()
        for doc in data["data"]:
            assert doc["series"] == 38

    def test_list_with_release_filter(self, client):
        resp = client.get("/api/v1/documents?release=R18")
        assert resp.status_code == 200

    def test_list_negative_offset(self, client):
        resp = client.get("/api/v1/documents?offset=-1")
        assert resp.status_code == 422

    def test_get_document(self, client):
        resp = client.get("/api/v1/documents/d1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["doc_id"] == "d1"

    def test_get_document_not_found(self, client):
        resp = client.get("/api/v1/documents/nonexistent")
        assert resp.status_code == 404

    def test_get_document_chunks(self, client):
        resp = client.get("/api/v1/documents/d1/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert len(data["data"]) > 0

    def test_chunks_not_found(self, client, mock_pipeline):
        mock_pipeline._store.get_document_chunks.return_value = []
        resp = client.get("/api/v1/documents/d1/chunks")
        assert resp.status_code == 404

    def test_delete_document(self, client):
        resp = client.delete("/api/v1/documents/d1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["doc_id"] == "d1"


# ── Stats 端点 ──


class TestStatsEndpoint:
    """GET /api/v1/stats"""

    def test_stats(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        stats = data["data"]
        assert stats["total_docs"] >= 0
        assert stats["total_chunks"] >= 0
        assert isinstance(stats["releases"], dict)
        assert isinstance(stats["series_distribution"], dict)
        assert "vector_db" in stats


# ── SSE Streaming 端点 ──


class TestAskStreamEndpoint:
    """POST /api/v1/ask/stream"""

    def test_stream_basic(self, client):
        resp = client.post("/api/v1/ask/stream", json={
            "query": "PDU session setup",
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # 读取 SSE 事件流
        body = resp.text
        assert 'data:' in body
        # 应包含 sources + chunks + done
        assert '"type":"sources"' in body or '"type": "sources"' in body
        assert '"type":"chunk"' in body or '"type": "chunk"' in body
        assert '"type":"done"' in body or '"type": "done"' in body
        # done 事件携带完整回答
        assert "TS 38.413" in body

    def test_stream_query_too_long(self, client):
        resp = client.post("/api/v1/ask/stream", json={
            "query": "x" * 2001,
        })
        assert resp.status_code == 422

    def test_stream_pipeline_error(self, client, mock_pipeline):
        def _boom(query, reranker_enabled=True, **kwargs):
            raise RuntimeError("Stream error")
            yield  # pragma: no cover
        mock_pipeline.ask_stream = _boom
        resp = client.post("/api/v1/ask/stream", json={"query": "test"})
        assert resp.status_code == 200
        # SSE 错误以 error 事件返回
        body = resp.text
        assert '"type":"error"' in body or '"type": "error"' in body
        assert "Stream error" in body


# ── 引用图谱端点 ──


class TestRefsGraphEndpoint:
    """GET /api/v1/refs/graph"""

    def test_refs_graph_mocked(self, client, mock_pipeline):
        """Mock Milvus 查询返回引用结果."""
        mock_pipeline._store._ensure_connected = MagicMock()
        mock_pipeline._store._collection = MagicMock()
        mock_pipeline._store._collection.query.return_value = [
            {"text": "Refer to TS 38.413 §8.3.1 for details",
             "spec_number": "38.300", "doc_id": "d1", "release": "R18"},
            {"text": "See also TS 23.501 §5.6.7",
             "spec_number": "38.300", "doc_id": "d2", "release": "R18"},
        ]

        resp = client.get("/api/v1/refs/graph?spec=38.300")
        assert resp.status_code == 200
        data = resp.json()
        assert data["spec"] == "38.300"
        assert "reference_count" in data
        assert "references" in data


# ── 请求模型验证 ──


class TestRequestValidation:
    """Pydantic 请求模型边界验证."""

    def test_ask_request_defaults(self, client):
        resp = client.post("/api/v1/ask", json={"query": "test"})
        assert resp.status_code == 200

    def test_search_request_minimal(self, client):
        resp = client.post("/api/v1/search", json={"query": "test"})
        assert resp.status_code == 200

    def test_ask_request_top_k_boundary_valid(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "test", "top_k": 1,
        })
        assert resp.status_code == 200

    def test_ask_request_top_k_zero(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "test", "top_k": 0,
        })
        assert resp.status_code == 422

    def test_ask_request_top_k_max(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "test", "top_k": 50,
        })
        assert resp.status_code == 200

    def test_ask_request_top_k_exceeds(self, client):
        resp = client.post("/api/v1/ask", json={
            "query": "test", "top_k": 51,
        })
        assert resp.status_code == 422


# ── API 响应格式一致性 ──


class TestAPIResponseFormat:
    """验证所有端点的 APIResponse 包装格式."""

    def test_search_has_api_response_structure(self, client):
        resp = client.post("/api/v1/search", json={"query": "test"})
        data = resp.json()
        assert "success" in data
        assert "data" in data
        assert "error" in data
        assert "meta" in data

    def test_stats_has_api_response_structure(self, client):
        resp = client.get("/api/v1/stats")
        data = resp.json()
        assert "success" in data
        assert "data" in data

    def test_documents_has_pagination_meta(self, client):
        resp = client.get("/api/v1/documents?offset=0&limit=10")
        data = resp.json()
        assert "meta" in data
        pagination = data["meta"].get("pagination", data["meta"])
        assert "offset" in pagination
        assert "limit" in pagination
        assert "total" in pagination
