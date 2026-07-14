"""auth.py 单元测试 — API Key 认证逻辑与常量."""

import importlib
import os


class TestSkipPaths:
    """_SKIP_PATHS — 无需认证的路径前缀."""

    def test_contains_health(self):
        from src.api.auth import _SKIP_PATHS
        assert "/api/v1/health" in _SKIP_PATHS

    def test_contains_docs(self):
        from src.api.auth import _SKIP_PATHS
        assert "/docs" in _SKIP_PATHS

    def test_contains_openapi(self):
        from src.api.auth import _SKIP_PATHS
        assert "/openapi.json" in _SKIP_PATHS

    def test_mcp_skipped(self):
        from src.api.auth import _SKIP_PATHS
        assert "/api/v1/mcp" in _SKIP_PATHS


class TestKeyParsing:
    """API_KEYS 环境变量解析."""

    def test_empty_env(self):
        # 重新加载模块以获取最新环境变量
        old = os.environ.get("API_KEYS", "")
        os.environ["API_KEYS"] = ""
        import src.api.auth as auth
        importlib.reload(auth)
        assert auth._VALID_KEYS == set()
        if old:
            os.environ["API_KEYS"] = old

    def test_single_key(self):
        old = os.environ.get("API_KEYS", "")
        os.environ["API_KEYS"] = "my-secret-key"
        import src.api.auth as auth
        importlib.reload(auth)
        assert auth._VALID_KEYS == {"my-secret-key"}
        if old:
            os.environ["API_KEYS"] = old

    def test_multiple_keys(self):
        old = os.environ.get("API_KEYS", "")
        os.environ["API_KEYS"] = "key1, key2 ,key3"
        import src.api.auth as auth
        importlib.reload(auth)
        assert auth._VALID_KEYS == {"key1", "key2", "key3"}
        if old:
            os.environ["API_KEYS"] = old

    def test_whitespace_only(self):
        old = os.environ.get("API_KEYS", "")
        os.environ["API_KEYS"] = "  , , "
        import src.api.auth as auth
        importlib.reload(auth)
        assert auth._VALID_KEYS == set()
        if old:
            os.environ["API_KEYS"] = old
