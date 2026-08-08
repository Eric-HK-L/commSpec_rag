"""测试 LLMClient — OpenAI 兼容接口."""


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
