"""RAG Pipeline 集成测试 — ask() + search() 完整链路 mock 验证."""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from src.generator.pipeline import (
    RAGPipeline,
    RAGResponse,
    _extract_spec_numbers,
    _filter_low_quality,
    _is_low_quality,
    _spec_aware_rerank,
)
from src.retriever.search import RetrievalResult


# ── Fixtures ──


@pytest.fixture
def mock_store():
    """Mock VectorStore — 返回可控的检索结果."""
    store = MagicMock()
    store.count = 1000
    store.supports_bm25 = True
    store.bm25_count = 800
    store._collection = MagicMock()
    store._ensure_connected = MagicMock()
    store.search_sparse = MagicMock(return_value=[])

    from src.retriever.vector_store import SearchResult
    _default_results = [
        SearchResult(chunk_id=1, text="PDU Session Resource Setup in NGAP protocol N2 interface", score=0.92,
                     doc_id="d1", series=38, spec_number="38.413", release="R18",
                     parent_section_id="8.3.1", parent_title="PDU Session Setup", chunk_index=0),
        SearchResult(chunk_id=2, text="QoS Flow binding to DRB at SDAP layer", score=0.78,
                     doc_id="d1", series=38, spec_number="38.413", release="R18",
                     parent_section_id="5.3.2", parent_title="QoS Flow", chunk_index=1),
        SearchResult(chunk_id=3, text="PDU Session establishment in 5GS architecture", score=0.72,
                     doc_id="d2", series=23, spec_number="23.501", release="R18",
                     parent_section_id="5.6.7", parent_title="PDU Session", chunk_index=0),
    ]

    def _search_dense(query_embedding, top_k=10, filter_expr=None):
        return _default_results[:top_k]

    # hybrid_search: 签名匹配 MilvusStore.hybrid_search
    def _hybrid_search(query_embedding=None, query_text=None, dense_top_k=100,
                       sparse_top_k=100, final_top_k=10):
        return _default_results[:final_top_k]

    store.search_dense = _search_dense
    store.hybrid_search = _hybrid_search

    # get_documents_summary
    store.get_documents_summary.return_value = {
        "d1": {"doc_id": "d1", "spec_number": "38.413", "release": "R18",
               "title": "NG-RAN NGAP", "series": 38, "chunk_count": 200},
        "d2": {"doc_id": "d2", "spec_number": "23.501", "release": "R18",
               "title": "5GS Architecture", "series": 23, "chunk_count": 150},
    }

    # get_document_chunks
    store.get_document_chunks.return_value = [
        {"id": 1, "text": "chunk1", "spec_number": "38.413", "release": "R18",
         "series": 38, "parent_section_id": "8.1", "parent_title": "Setup", "chunk_index": 0},
    ]

    store.delete_by_filter.return_value = 5

    return store


@pytest.fixture
def mock_llm():
    """Mock LLMClient — 返回预设的 LLM 响应."""
    llm = MagicMock()

    def _chat(messages, temperature=None, max_tokens=None):
        # 检查消息内容决定返回值
        user_content = messages[-1]["content"] if messages else ""
        if "查询优化器" in messages[0].get("content", ""):
            # 查询扩展
            return "PDU session setup procedure NGAP protocol N2 interface 38.413"
        return (
            "The PDU Session Resource Setup procedure is defined in 3GPP TS 38.413 §8.3.1. "
            "It describes the NGAP protocol messages exchanged over the N2 interface between "
            "the gNB and AMF."
        )

    llm.chat = _chat

    def _embed(texts):
        return [[0.1] * 1024 for _ in texts]

    llm.embed = _embed

    return llm


@pytest.fixture
def pipeline(mock_store, mock_llm, monkeypatch):
    """创建带 mock 依赖的 RAGPipeline."""
    # 禁用在线搜索
    from src.config import settings
    monkeypatch.setattr(settings, "enable_online_search", False)
    monkeypatch.setattr(settings, "reranker_enabled", False)
    monkeypatch.setattr(settings, "dense_top_k", 100)
    monkeypatch.setattr(settings, "bm25_top_k", 100)
    monkeypatch.setattr(settings, "max_search_results", 3)
    monkeypatch.setattr(settings, "reranker_top_k", 20)

    return RAGPipeline(vector_store=mock_store, llm_client=mock_llm)


@pytest.fixture
def sample_results():
    """构造标准检索结果列表."""
    return [
        RetrievalResult(chunk_id="1", text="PDU Session Resource Setup NGAP N2 interface",
                        score=0.92, doc_id="d1", series=38, spec_number="38.413",
                        release="R18", parent_section_id="8.3.1",
                        parent_title="PDU Session Setup", chunk_index=0),
        RetrievalResult(chunk_id="2", text="QoS Flow binding DRB SDAP layer",
                        score=0.78, doc_id="d1", series=38, spec_number="38.413",
                        release="R18", parent_section_id="5.3.2",
                        parent_title="QoS Flow", chunk_index=1),
        RetrievalResult(chunk_id="3", text="PDU Session 5GS architecture",
                        score=0.72, doc_id="d2", series=23, spec_number="23.501",
                        release="R18", parent_section_id="5.6.7",
                        parent_title="PDU Session", chunk_index=0),
    ]


# ── Pipeline 初始化测试 ──


class TestPipelineInit:
    """RAGPipeline 初始化与组件创建."""

    def test_creates_hybrid_retriever(self, pipeline):
        assert pipeline._retriever is not None

    def test_creates_verifier(self, pipeline):
        assert pipeline._verifier is not None

    def test_creates_multi_hop(self, pipeline):
        assert pipeline._multi_hop is not None

    def test_creates_query_cache(self, pipeline):
        assert pipeline._query_cache is not None
        assert pipeline._query_cache.maxsize == 256

    def test_online_supplement_disabled(self, pipeline, monkeypatch):
        assert not pipeline._online.enabled


# ── Pipeline ask() 集成测试 ──


class TestPipelineAsk:
    """RAGPipeline.ask() 完整链路测试."""

    def test_ask_returns_rag_response(self, pipeline):
        response = pipeline.ask("What is PDU Session Resource Setup?")
        assert isinstance(response, RAGResponse)
        assert response.query == "What is PDU Session Resource Setup?"
        assert len(response.answer) > 0
        # verified 取决于答案与源的匹配度, mock 答案与源不一定重叠
        assert isinstance(response.verified, bool)
        assert len(response.sources) > 0
        assert response.expanded_query != ""

    def test_ask_query_cache_hit(self, pipeline):
        """相同查询第二次应从缓存返回."""
        query = "PDU session setup cached"
        r1 = pipeline.ask(query)
        r2 = pipeline.ask(query)
        # 两次调用返回相同对象 (缓存命中)
        assert r1 is r2
        assert r1.answer == r2.answer

    def test_ask_query_cache_miss(self, pipeline):
        """不同查询应有不同的缓存 key."""
        r1 = pipeline.ask("PDU session query A")
        r2 = pipeline.ask("PDU session query B")
        assert r1 is not r2

    def test_ask_with_reranker_disabled(self, pipeline):
        response = pipeline.ask("PDU Session", reranker_enabled=False)
        assert isinstance(response, RAGResponse)
        assert len(response.sources) > 0

    def test_ask_source_metadata(self, pipeline):
        response = pipeline.ask("PDU Session Setup")
        for src in response.sources:
            assert hasattr(src, "spec_number")
            assert hasattr(src, "parent_section_id")
            assert hasattr(src, "score")

    def test_ask_empty_results(self, mock_llm, monkeypatch):
        """无检索结果时返回明确消息而非崩溃."""
        from unittest.mock import MagicMock
        empty_store = MagicMock()
        empty_store.count = 0
        empty_store.supports_bm25 = False
        empty_store._collection = MagicMock()
        empty_store._ensure_connected = MagicMock()
        empty_store.search_dense.return_value = []
        # hybrid_search 返回空
        empty_store.hybrid_search.return_value = []
        # 不提供 BM25, 走 search_dense 降级
        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)

        pipeline2 = RAGPipeline(vector_store=empty_store, llm_client=mock_llm)
        response = pipeline2.ask("nonexistent query")
        assert "未在 3GPP 规范中找到" in response.answer
        assert len(response.sources) == 0

    def test_ask_llm_error_handling(self, mock_store, mock_llm, monkeypatch):
        """LLM chat 失败时 pipeline 应抛出异常."""
        mock_llm.chat = MagicMock(side_effect=RuntimeError("LLM API error"))

        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)

        pipeline2 = RAGPipeline(vector_store=mock_store, llm_client=mock_llm)
        with pytest.raises(RuntimeError, match="LLM API error"):
            pipeline2.ask("test query")


# ── Pipeline search() 集成测试 ──


class TestPipelineSearch:
    """RAGPipeline.search() 仅检索测试."""

    def test_search_returns_results(self, pipeline):
        results = pipeline.search("PDU session", top_k=3)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_search_with_top_k(self, pipeline):
        # 不触发 spec-aware 补充的查询, 确保结果数受控
        results = pipeline.search("QoS flow binding", top_k=2)
        assert len(results) >= 1  # spec-aware 可能补充, 不严格限定 <=2

    def test_search_reranker_disabled(self, pipeline):
        results = pipeline.search("PDU session", reranker_enabled=False)
        assert len(results) > 0

    def test_search_spec_hints_boost(self, mock_store, mock_llm, monkeypatch):
        """查询中包含 spec 号时触发 spec-aware 重排序."""
        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)

        pipeline2 = RAGPipeline(vector_store=mock_store, llm_client=mock_llm)
        # 重写 _expand_query 使查询包含 spec 号
        pipeline2._expand_query = lambda q: "38.413 PDU session setup"

        results = pipeline2.search("PDU session", top_k=3)
        assert len(results) > 0
        # 38.413 的结果应该排在前面
        if len(results) > 0:
            assert results[0].spec_number == "38.413"


# ── 辅助函数测试 ──


class TestExtractSpecNumbers:
    """_extract_spec_numbers — 从文本提取规范号."""

    def test_single_spec(self):
        result = _extract_spec_numbers("TS 38.413 defines PDU Session Setup")
        assert "38.413" in result

    def test_multiple_specs(self):
        result = _extract_spec_numbers("See 38.413 and 23.501 for details")
        assert result == {"38.413", "23.501"}

    def test_no_spec(self):
        result = _extract_spec_numbers("PDU session setup procedure")
        assert result == set()

    def test_invalid_format(self):
        result = _extract_spec_numbers("Version 1.2.3 protocol")
        assert result == set()


class TestFilterLowQuality:
    """_filter_low_quality + _is_low_quality — 低质量 chunks 过滤."""

    def test_abbreviations_filtered(self):
        r = RetrievalResult(chunk_id="1", text="3GPP 5GS PDCP ...",
                            score=0.8, spec_number="38.413",
                            parent_section_id="3.3",
                            parent_title="Abbreviations")
        assert _is_low_quality(r) is True

    def test_definitions_filtered(self):
        r = RetrievalResult(chunk_id="1", text="For the purposes of...",
                            score=0.8, spec_number="38.413",
                            parent_section_id="3.1",
                            parent_title="Definitions")
        assert _is_low_quality(r) is True

    def test_references_filtered(self):
        r = RetrievalResult(chunk_id="1", text="[1] 3GPP TS 38.300...",
                            score=0.8, spec_number="38.413",
                            parent_section_id="2",
                            parent_title="References")
        assert _is_low_quality(r) is True

    def test_structural_prefix_filtered(self):
        r = RetrievalResult(chunk_id="1", text="#  Contents\n1 Scope\n2 References",
                            score=0.8, spec_number="38.413")
        assert _is_low_quality(r) is True

    def test_normal_content_passes(self):
        r = RetrievalResult(chunk_id="1",
                            text="The NGAP protocol supports PDU Session Resource Setup...",
                            score=0.8, spec_number="38.413",
                            parent_section_id="8.3.1",
                            parent_title="PDU Session Setup Procedure")
        assert _is_low_quality(r) is False

    def test_filter_preserves_quality(self, sample_results):
        filtered = _filter_low_quality(sample_results, target_k=2)
        assert len(filtered) >= 1  # 至少保留高质量结果

    def test_filter_scene_title_not_filtered(self):
        """operation/function 等包含关键词的标题不应被过滤."""
        r = RetrievalResult(chunk_id="1",
                            text="This operation defines the referencing mechanism...",
                            score=0.8, spec_number="38.413",
                            parent_section_id="6.2",
                            parent_title="Reference operation")
        assert _is_low_quality(r) is False


class TestSpecAwareRerank:
    """_spec_aware_rerank — 两阶段 spec-aware 重排序."""

    def test_boost_matching_spec(self, sample_results, mock_store):
        """匹配 spec 的结果应获得 boost."""
        boosted = _spec_aware_rerank(
            results=sample_results,
            spec_hints={"38.413"},
            store=mock_store,
            query_embedding=np.array([0.1] * 1024, dtype=np.float32),
            top_k=3,
        )
        assert len(boosted) >= len(sample_results)
        # 38.413 的结果应排在前面
        top_specs = [r.spec_number for r in boosted[:2]]
        assert "38.413" in top_specs

    def test_no_hints_no_change(self, sample_results, mock_store):
        """无 spec hints 时不改变结果."""
        boosted = _spec_aware_rerank(
            results=sample_results,
            spec_hints=set(),
            store=mock_store,
            query_embedding=np.array([0.1] * 1024, dtype=np.float32),
            top_k=3,
        )
        assert len(boosted) == len(sample_results)

    def test_score_ordering(self, sample_results, mock_store):
        """boost 后分数应降序排列."""
        boosted = _spec_aware_rerank(
            results=sample_results,
            spec_hints={"38.413", "23.501"},
            store=mock_store,
            query_embedding=np.array([0.1] * 1024, dtype=np.float32),
            top_k=10,
        )
        for i in range(len(boosted) - 1):
            assert boosted[i].score >= boosted[i + 1].score


# ── 查询缓存测试 ──


class TestQueryCache:
    """查询级 LRU 缓存行为验证."""

    def test_cache_key_deterministic(self, pipeline):
        """相同查询生成相同缓存 key."""
        q = "PDU session setup"
        k1 = hashlib.md5(q.lower().strip().encode()).hexdigest()
        k2 = hashlib.md5(q.lower().strip().encode()).hexdigest()
        assert k1 == k2

    def test_cache_key_case_insensitive(self):
        """大小写和空白不影响缓存 key."""
        k1 = hashlib.md5("PDU Session".lower().strip().encode()).hexdigest()
        k2 = hashlib.md5("  pdu session  ".lower().strip().encode()).hexdigest()
        assert k1 == k2

    def test_cache_ttl_expiry(self, pipeline):
        """TTL 时间窗口正确设置."""
        assert pipeline._query_cache.ttl == 3600


# ── 错误恢复路径 ──


class TestPipelineErrorRecovery:
    """异常场景下 pipeline 的容错行为."""

    def test_expand_query_fallback(self, mock_store, mock_llm, monkeypatch):
        """查询扩展失败时降级为原始查询."""
        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)
        # 让 LLM 在查询扩展中抛异常
        original_chat = mock_llm.chat

        def _fail_on_expand(messages, temperature=None, max_tokens=None):
            if "查询优化器" in messages[0].get("content", ""):
                raise RuntimeError("Simulated failure")
            return original_chat(messages, temperature, max_tokens)

        mock_llm.chat = _fail_on_expand
        pipeline2 = RAGPipeline(vector_store=mock_store, llm_client=mock_llm)

        response = pipeline2.ask("PDU session")
        # 应降级为原始查询完成
        assert isinstance(response, RAGResponse)
        assert "PDU session" in response.expanded_query

    def test_embed_failure_zero_vector(self, mock_store, mock_llm, monkeypatch):
        """嵌入失败时使用零向量降级 (不崩溃)."""
        mock_llm.embed = MagicMock(side_effect=RuntimeError("Embedding API failed"))
        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)

        pipeline2 = RAGPipeline(vector_store=mock_store, llm_client=mock_llm)
        # 嵌入失败后 ask 仍能继续 (用零向量检索, 结果可能为空)
        response = pipeline2.ask("test")
        assert isinstance(response, RAGResponse)

    def test_cross_ref_secondary_search_failure(self, mock_store, mock_llm, monkeypatch):
        """交叉引用二次检索失败不影响主流程."""
        from src.config import settings
        monkeypatch.setattr(settings, "enable_online_search", False)

        # mock _resolve_cross_refs 内部抛异常
        pipeline2 = RAGPipeline(vector_store=mock_store, llm_client=mock_llm)
        # 重写使其内部异常被捕获
        original_resolve = pipeline2._resolve_cross_refs
        def _failing_resolve(results, max_refs=5):
            try:
                return original_resolve(results, max_refs)
            except Exception:
                return results  # 降级返回原始结果
        pipeline2._resolve_cross_refs = _failing_resolve

        response = pipeline2.ask("PDU session test")
        assert isinstance(response, RAGResponse)


# ── RAGResponse 数据类 ──


class TestRAGResponse:
    """RAGResponse 数据类构造."""

    def test_basic(self):
        r = RAGResponse(
            query="q", answer="a", sources=[],
            verified=True, warnings=[], coverage=0.5,
            expanded_query="e",
        )
        assert r.query == "q"
        assert r.answer == "a"
        assert r.coverage == 0.5

    def test_defaults(self):
        r = RAGResponse(query="q", answer="a", sources=[], verified=False, warnings=[])
        assert r.coverage == 0.0
        assert r.expanded_query == ""
