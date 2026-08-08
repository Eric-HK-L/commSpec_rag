"""查询缓存 Key 单元测试 — 必须包含所有影响结果的参数."""

from src.generator.pipeline import _build_cache_key


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
