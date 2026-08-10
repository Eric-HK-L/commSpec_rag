"""查询缓存 Key 单元测试 — 必须包含所有影响结果的参数."""

from unittest.mock import MagicMock

from src.generator.pipeline import (
    FALLBACK_ANSWER,
    NOT_FOUND_ANSWER,
    RAGPipeline,
    RAGResponse,
    _build_cache_key,
    _is_cacheable,
)
from src.retriever.planner import RetrievalContext
from src.retriever.search import RetrievalResult


class TestBuildCacheKey:

    def test_same_query_same_key(self):
        assert _build_cache_key("PDU session") == _build_cache_key("PDU session")

    def test_case_and_whitespace_normalized(self):
        assert _build_cache_key("  PDU Session  ") == _build_cache_key("pdu session")

    def test_release_changes_key(self):
        assert _build_cache_key("x", release="R18") != _build_cache_key("x", release="R17")

    def test_series_changes_key(self):
        assert _build_cache_key("x", series="38") != _build_cache_key("x", series="23")

    def test_doc_type_changes_key(self):
        assert _build_cache_key("x", doc_type="3gpp") != _build_cache_key("x", doc_type="oran")

    def test_reranker_flag_changes_key(self):
        assert (
            _build_cache_key("x", reranker_enabled=True)
            != _build_cache_key("x", reranker_enabled=False)
        )

    def test_history_changes_key(self):
        h1 = [{"role": "user", "content": "继续解释"}]
        h2 = [{"role": "user", "content": "换个话题"}]
        assert _build_cache_key("x", history=h1) != _build_cache_key("x", history=h2)

    def test_history_order_matters(self):
        h1 = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        h2 = [{"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        assert _build_cache_key("x", history=h1) != _build_cache_key("x", history=h2)

    def test_deterministic_across_calls(self):
        k1 = _build_cache_key("q", release="R18", series="38", history=[{"role": "user", "content": "hi"}])
        k2 = _build_cache_key("q", release="R18", series="38", history=[{"role": "user", "content": "hi"}])
        assert k1 == k2


def _resp(answer: str) -> RAGResponse:
    return RAGResponse(
        query="q", answer=answer, sources=[], verified=True, warnings=[],
    )


class TestIsCacheable:
    """空/兜底回答不写缓存 — DeepSeek 空响应导致的 '抱歉' 文案不应被缓存 1 小时."""

    def test_normal_answer_cacheable(self):
        assert _is_cacheable(_resp("SSB consists of 4 OFDM symbols."))

    def test_empty_answer_not_cacheable(self):
        assert not _is_cacheable(_resp(""))

    def test_whitespace_answer_not_cacheable(self):
        assert not _is_cacheable(_resp("   \n  "))

    def test_fallback_answer_not_cacheable(self):
        assert not _is_cacheable(_resp(FALLBACK_ANSWER))

    def test_not_found_answer_not_cacheable(self):
        assert not _is_cacheable(_resp(NOT_FOUND_ANSWER))


class TestPipelineCacheWrites:
    """ask()/ask_stream() 兜底回答不写入查询缓存."""

    def _make_pipeline(self) -> RAGPipeline:
        pipe = RAGPipeline(vector_store=MagicMock())
        pipe._query_cache.clear()
        return pipe

    def _mock_context(self) -> MagicMock:
        ctx = RetrievalContext(
            query_lang="zh", search_query="q", expanded_query="q",
            results=[RetrievalResult(
                chunk_id=1, text="ctx", score=0.9, spec_number="38300",
                parent_section_id="s1", parent_title="t",
            )],
            release_note="", online_context="",
        )
        mock = MagicMock(return_value=ctx)
        return mock

    def test_ask_fallback_not_cached(self):
        pipe = self._make_pipeline()
        pipe._retrieve_context = self._mock_context()
        pipe._llm = MagicMock()
        pipe._llm.chat.return_value = ""  # LLM 返回空 → 触发兜底文案

        resp = pipe.ask("测试问题")
        assert resp.answer == FALLBACK_ANSWER
        assert len(pipe._query_cache) == 0  # 兜底回答不缓存

    def test_ask_normal_cached(self):
        pipe = self._make_pipeline()
        pipe._retrieve_context = self._mock_context()
        pipe._llm = MagicMock()
        pipe._llm.chat.return_value = "SSB 由 PSS/SSS/PBCH/DM-RS 组成。"

        resp = pipe.ask("测试问题")
        assert resp.answer == "SSB 由 PSS/SSS/PBCH/DM-RS 组成。"
        assert len(pipe._query_cache) == 1  # 正常回答缓存

    def test_ask_stream_fallback_not_cached(self):
        pipe = self._make_pipeline()
        pipe._retrieve_context = self._mock_context()
        pipe._llm = MagicMock()
        pipe._llm.chat_stream.return_value = iter([])  # 空流
        pipe._llm.chat.return_value = ""  # 非流式重试也空 → 兜底

        events = list(pipe.ask_stream("测试问题"))
        assert events[-1][1]["answer"] == FALLBACK_ANSWER
        assert len(pipe._query_cache) == 0  # 兜底回答不缓存
