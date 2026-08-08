"""集成测试层 (BT) — 真实模块编排 + mock 外部服务 (Milvus/LLM/网络)."""
import pytest

pytestmark = pytest.mark.bt
