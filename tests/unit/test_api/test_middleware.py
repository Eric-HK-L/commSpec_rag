"""middleware.py 单元测试 — 速率限制与全局异常处理."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.rest.middleware import RateLimitMiddleware, register_exception_handlers


def _make_limited_app(rpm: int = 2, enabled: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rpm=rpm, enabled=enabled)

    @app.get("/api/v1/ask")
    def ask():
        return {"ok": True}

    @app.get("/api/v1/search")
    def search():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


class TestRateLimitMiddleware:

    def test_limited_after_rpm_exceeded(self):
        client = TestClient(_make_limited_app(rpm=2))
        assert client.get("/api/v1/ask").status_code == 200
        assert client.get("/api/v1/ask").status_code == 200
        assert client.get("/api/v1/ask").status_code == 429

    def test_search_also_limited(self):
        client = TestClient(_make_limited_app(rpm=1))
        assert client.get("/api/v1/search").status_code == 200
        assert client.get("/api/v1/search").status_code == 429

    def test_health_not_limited(self):
        client = TestClient(_make_limited_app(rpm=1))
        client.get("/api/v1/ask")
        assert client.get("/health").status_code == 200

    def test_disabled_passes_through(self):
        client = TestClient(_make_limited_app(rpm=1, enabled=False))
        for _ in range(5):
            assert client.get("/api/v1/ask").status_code == 200


def _make_error_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail /etc/passwd")

    @app.get("/bad-value")
    def bad_value():
        raise ValueError("参数不合法")

    @app.get("/missing-file")
    def missing_file():
        raise FileNotFoundError("no such file")

    return app


class TestExceptionHandlers:

    def test_generic_error_hides_internal_detail(self):
        client = TestClient(_make_error_app(), raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_code"] == "INTERNAL_ERROR"
        assert "secret" not in resp.text
        assert "passwd" not in resp.text

    def test_value_error_maps_to_400(self):
        client = TestClient(_make_error_app(), raise_server_exceptions=False)
        resp = client.get("/bad-value")
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "INVALID_PARAMETER"

    def test_file_not_found_maps_to_404(self):
        client = TestClient(_make_error_app(), raise_server_exceptions=False)
        resp = client.get("/missing-file")
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "NOT_FOUND"
