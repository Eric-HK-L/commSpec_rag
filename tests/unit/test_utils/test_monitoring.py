"""monitoring.py 单元测试 — _route_label 路径归一化."""

from unittest.mock import MagicMock

from src.utils.monitoring import _route_label


class TestRouteLabel:
    """_route_label — 将 HTTP 路径转为稳定的 Prometheus 标签."""

    def test_ask_endpoint(self):
        request = MagicMock()
        request.url.path = "/api/v1/ask"
        request.scope = {}
        assert _route_label(request) == "ask"

    def test_search_endpoint(self):
        request = MagicMock()
        request.url.path = "/api/v1/search"
        request.scope = {}
        assert _route_label(request) == "search"

    def test_documents_endpoint(self):
        request = MagicMock()
        request.url.path = "/api/v1/documents"
        request.scope = {}
        assert _route_label(request) == "documents"

    def test_health_endpoint(self):
        request = MagicMock()
        request.url.path = "/api/v1/health"
        request.scope = {}
        assert _route_label(request) == "health"

    def test_mcp_endpoint(self):
        request = MagicMock()
        request.url.path = "/api/v1/mcp/tools/call"
        request.scope = {}
        assert _route_label(request) == "mcp"

    def test_root(self):
        request = MagicMock()
        request.url.path = "/"
        request.scope = {}
        assert _route_label(request) == "root"

    def test_unknown_path(self):
        request = MagicMock()
        request.url.path = "/random/path"
        request.scope = {}
        assert _route_label(request) == "random_path"

    def test_metrics_excluded_handled(self):
        # metrics 端点本身不走中间件，但 _route_label 仍应能解析
        request = MagicMock()
        request.url.path = "/metrics"
        request.scope = {}
        assert _route_label(request) == "metrics"
