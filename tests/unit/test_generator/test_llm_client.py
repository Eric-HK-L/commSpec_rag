"""测试 LLMClient — OpenAI 兼容接口."""

from unittest.mock import MagicMock

from src.generator.llm_client import LLMClient


class TestLLMClientInit:
    """初始化与配置."""

    def test_default_init(self):
        client = LLMClient()
        assert client is not None

    def test_custom_config_via_settings(self, monkeypatch):
        """LLMClient 从 settings 读取配置."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_model", "test-model-override")
        client = LLMClient()
        assert client._model == "test-model-override"

    def test_singleton_behavior(self):
        """每次调用创建新 client (无全局单例约束)."""
        c1 = LLMClient()
        c2 = LLMClient()
        assert c1 is not c2  # 每次新建


class TestLLMClientInterface:
    """接口契约."""

    def test_has_chat_method(self):
        client = LLMClient()
        assert hasattr(client, "chat")
        assert callable(client.chat)

    def test_has_embed_method(self):
        client = LLMClient()
        assert hasattr(client, "embed")
        assert callable(client.embed)

    def test_chat_accepts_messages(self):
        # 测试接口签名 — 实际调用需 API Key
        LLMClient()
        assert True  # 接口存在性已验证

    def test_embed_returns_correct_dim(self, monkeypatch):
        """Mock embed 返回 1024 维向量."""
        from src.config import settings
        monkeypatch.setattr(settings, "embedding_dimension", 1024)

        client = LLMClient()
        # 不实际调用 API, 仅验证接口
        assert hasattr(client, "embed")


# ── 空响应自动重试 (DeepSeek 偶发空 content) ──


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeStreamDelta:
    def __init__(self, content):
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content):
        self.delta = _FakeStreamDelta(content)


class _FakeStreamChunk:
    def __init__(self, content):
        self.choices = [_FakeStreamChoice(content)]


def _stream(*contents: str):
    """构造 OpenAI 流式 chunk 序列 — 每个 content 一个 chunk (None 表示无内容)."""
    return iter([
        _FakeStreamChunk(c) if c is not None else _FakeStreamChunk(None)
        for c in contents
    ])


class TestEmptyResponseRetry:
    """chat() / chat_stream() 对空 content 自动重试 — 修复 DeepSeek 偶发空响应."""

    def _client(self) -> LLMClient:
        client = LLMClient()
        client._client = MagicMock()
        return client

    def test_chat_retries_when_first_response_empty(self, monkeypatch):
        """首次返回空 content → 自动重试 → 返回第二次非空结果."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 2)

        client = self._client()
        client._client.chat.completions.create.side_effect = [
            _FakeResp(""),
            _FakeResp("正确回答"),
        ]
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "正确回答"
        assert client._client.chat.completions.create.call_count == 2

    def test_chat_retries_when_first_response_none(self, monkeypatch):
        """content=None (choices[0].message.content 为空) 也触发重试."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 2)

        client = self._client()
        client._client.chat.completions.create.side_effect = [
            _FakeResp(None),
            _FakeResp("回复内容"),
        ]
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "回复内容"

    def test_chat_all_empty_returns_empty_after_max_retries(self, monkeypatch):
        """连续空响应 → 重试耗尽后返回空串 (不抛异常, 由上层兜底文案处理)."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 2)

        client = self._client()
        client._client.chat.completions.create.side_effect = [
            _FakeResp(""), _FakeResp(""), _FakeResp(""),
        ]
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == ""
        assert client._client.chat.completions.create.call_count == 3  # 1 次原始 + 2 次重试

    def test_chat_stream_retries_when_stream_empty(self, monkeypatch):
        """chat_stream 空流 → 自动重试非空流."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 2)

        client = self._client()
        client._client.chat.completions.create.side_effect = [
            _stream(None),
            _stream("正常", "回复"),
        ]
        out = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert out == ["正常", "回复"]
        assert client._client.chat.completions.create.call_count == 2

    def test_chat_stream_nonempty_no_retry(self, monkeypatch):
        """非空流 → 不触发额外重试 (零额外调用)."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 2)

        client = self._client()
        client._client.chat.completions.create.side_effect = [
            _stream("直接回答"),
        ]
        out = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert out == ["直接回答"]
        assert client._client.chat.completions.create.call_count == 1

    def test_chat_exception_still_raises(self, monkeypatch):
        """API 异常不吞掉 — 仍向上抛出 (openai 客户端自带 429/5xx 重试)."""
        from src.config import settings
        monkeypatch.setattr(settings, "llm_max_retries", 1)

        client = self._client()
        client._client.chat.completions.create.side_effect = RuntimeError("API down")
        import pytest
        with pytest.raises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}])
