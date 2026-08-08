"""管理后台认证集成测试 — 登录/登出/受保护端点/篡改 Cookie."""

import os

import pytest
from fastapi.testclient import TestClient

from src.config import settings

LOGIN_URL = "/api/v1/admin/login"
LOGOUT_URL = "/api/v1/admin/logout"
CONFIG_URL = "/api/v1/admin/config"


@pytest.fixture
def client(monkeypatch):
    """构造带真实认证配置的 FastAPI TestClient (不触发 lifespan, 无 Milvus)."""
    os.environ["API_KEYS"] = ""
    monkeypatch.setattr(settings, "admin_username", "admin")
    monkeypatch.setattr(settings, "admin_password", "test-pass-123")
    monkeypatch.setattr(settings, "admin_session_secret", "unit-test-secret")
    monkeypatch.setattr(settings, "admin_session_ttl_hours", 12)
    monkeypatch.setattr(settings, "rate_limit_enabled", False)

    # 兜底: 无论模块加载顺序如何, 确保 API Key 校验不干扰本测试
    import src.api.auth as auth
    from src.main import app
    auth._VALID_KEYS.clear()

    return TestClient(app)


class TestAdminLogin:

    def test_login_success_sets_session_cookie(self, client):
        resp = client.post(LOGIN_URL, json={"username": "admin", "password": "test-pass-123"})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert client.cookies.get("admin_session")

    def test_login_wrong_password_rejected(self, client):
        resp = client.post(LOGIN_URL, json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 401

    def test_login_wrong_username_rejected(self, client):
        resp = client.post(LOGIN_URL, json={"username": "root", "password": "test-pass-123"})
        assert resp.status_code == 401

    def test_login_disabled_without_password(self, client, monkeypatch):
        monkeypatch.setattr(settings, "admin_password", "")
        resp = client.post(LOGIN_URL, json={"username": "admin", "password": "test-pass-123"})
        assert resp.status_code == 403

    def test_login_missing_fields_rejected(self, client):
        resp = client.post(LOGIN_URL, json={"username": "admin"})
        assert resp.status_code == 422


class TestAdminProtection:

    def test_admin_endpoint_requires_login(self, client):
        resp = client.get(CONFIG_URL)
        assert resp.status_code == 401

    def test_admin_endpoint_ok_with_session(self, client):
        client.post(LOGIN_URL, json={"username": "admin", "password": "test-pass-123"})
        resp = client.get(CONFIG_URL)
        assert resp.status_code == 200

    def test_tampered_cookie_rejected(self, client):
        client.post(LOGIN_URL, json={"username": "admin", "password": "test-pass-123"})
        token = client.cookies.get("admin_session")
        # 翻转签名最后一个 hex 字符 — 必然与原始 token 不同, 避免旧逻辑
        # (翻转 token[-1] 后覆盖 token[-4]) 在字符恰好相同时篡改无效的 flaky
        last = token[-1]
        forged = token[:-1] + ("0" if last != "0" else "1")
        client.cookies.set("admin_session", forged)
        resp = client.get(CONFIG_URL)
        assert resp.status_code == 401

    def test_logout_clears_session(self, client):
        client.post(LOGIN_URL, json={"username": "admin", "password": "test-pass-123"})
        resp = client.post(LOGOUT_URL)
        assert resp.status_code == 200
        resp = client.get(CONFIG_URL)
        assert resp.status_code == 401
