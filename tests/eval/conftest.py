"""评测测试层 (eval) — 纯单元测试, 无外部依赖 (Milvus/LLM 全部 mock).

与 tests/unit 一致: conftest 是 pytest 插件, 不在节点树上,
这里用显式 collection hook 给目录内所有测试打 `unit` 标记, 供 `-m unit` 选择.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.unit)
