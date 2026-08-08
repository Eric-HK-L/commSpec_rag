"""集成测试层 (BT) — 真实模块编排 + mock 外部服务 (Milvus/LLM/网络).

conftest 是 pytest 插件, 不在节点树上, `pytestmark` 不会传播到测试节点.
这里用显式 collection hook 给目录内所有测试打 `bt` 标记, 供 `-m bt` 选择.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.bt)
