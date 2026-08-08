"""单元测试层 (UT) — 快速、无外部依赖，外部服务全部 mock.

conftest 是 pytest 插件, 不在节点树上, `pytestmark` 不会传播到测试节点.
这里用显式 collection hook 给目录内所有测试打 `unit` 标记, 供 `-m unit` 选择.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.unit)
