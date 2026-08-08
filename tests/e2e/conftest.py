"""端到端测试层 (E2E) — 需要真实 Milvus / 完整服务栈.

conftest 是 pytest 插件, 不在节点树上, `pytestmark` 不会传播到测试节点.
这里用显式 collection hook 给目录内所有测试打 `e2e` 标记, 供 `-m e2e` 选择.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.e2e)
