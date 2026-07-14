"""online_supplement.py 单元测试 — 触发条件判断与结果格式化（纯函数）."""

from src.retriever.online_supplement import (
    GoogleSearchProvider,
    OnlineResult,
    OnlineSupplement,
    TSpecLLMProvider,
)


class TestGoogleSearchProviderEnabled:

    def test_enabled_with_keys(self):
        p = GoogleSearchProvider(api_key="sk-fake", cse_id="abc123")
        assert p.enabled is True

    def test_disabled_no_keys(self):
        p = GoogleSearchProvider()
        assert p.enabled is False

    def test_disabled_partial(self):
        p = GoogleSearchProvider(api_key="sk-fake", cse_id="")
        assert p.enabled is False


class TestTSpecLLMProviderEnabled:

    def test_enabled_with_url(self):
        p = TSpecLLMProvider(base_url="https://example.com")
        assert p.enabled is True

    def test_disabled_no_url(self):
        p = TSpecLLMProvider()
        assert p.enabled is False


class TestShouldSupplement:

    def test_disabled_never_supplements(self):
        s = OnlineSupplement()  # 未配置任何 provider
        assert s.should_supplement(0.9, 10) is False
        assert s.should_supplement(0.0, 0) is False

    def test_zero_results_triggers(self):
        s = OnlineSupplement(google_api_key="k", google_cse_id="c")
        assert s.should_supplement(0.0, 0) is True

    def test_low_score_triggers(self):
        s = OnlineSupplement(google_api_key="k", google_cse_id="c", score_threshold=0.6)
        assert s.should_supplement(0.3, 5) is True

    def test_high_score_no_trigger(self):
        s = OnlineSupplement(google_api_key="k", google_cse_id="c", score_threshold=0.6)
        assert s.should_supplement(0.85, 10) is False

    def test_few_results_triggers(self):
        s = OnlineSupplement(google_api_key="k", google_cse_id="c", count_threshold=5)
        assert s.should_supplement(0.8, 2) is True

    def test_enough_results_no_trigger(self):
        s = OnlineSupplement(google_api_key="k", google_cse_id="c")
        assert s.should_supplement(0.8, 10) is False


class TestFormatAsContext:

    def test_empty_results(self):
        s = OnlineSupplement()
        assert s.format_as_context([]) == ""

    def test_single_result(self):
        s = OnlineSupplement()
        results = [OnlineResult(
            title="TS 38.413 Overview",
            snippet="This document specifies NGAP.",
            url="https://www.3gpp.org/ftp/Specs/archive/38_series/38.413/",
            source="google",
        )]
        ctx = s.format_as_context(results)
        assert "## 在线补充参考" in ctx
        assert "TS 38.413" in ctx
        assert "NGAP" in ctx

    def test_multiple_results(self):
        s = OnlineSupplement()
        results = [
            OnlineResult(title="TS 38.413", snippet="NGAP protocol", url="url1", source="google"),
            OnlineResult(title="TS 23.501", snippet="5G Architecture", url="url2", source="tspec-llm"),
        ]
        ctx = s.format_as_context(results)
        assert "### 参考 1" in ctx
        assert "### 参考 2" in ctx
        assert "url1" in ctx
        assert "url2" in ctx

    def test_external_source_label(self):
        s = OnlineSupplement()
        results = [OnlineResult(title="T", snippet="S", url="U", source="google")]
        ctx = s.format_as_context(results)
        assert "外部来源" in ctx
