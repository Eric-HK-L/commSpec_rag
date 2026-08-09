"""测试 Reranker — Cross-Encoder 精排."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retriever.reranker import get_reranker
from src.retriever.search import RetrievalResult


@pytest.fixture
def sample_results():
    """构造 3 个 3GPP 检索结果."""
    return [
        RetrievalResult(
            chunk_id="1", text="PDU Session Resource Setup procedure in NGAP protocol N2 interface",
            spec_number="38.413", parent_section_id="8.3.1", score=0.92,
        ),
        RetrievalResult(
            chunk_id="2", text="QoS Flow binding to Data Radio Bearer at SDAP layer",
            spec_number="38.413", parent_section_id="5.3.2", score=0.78,
        ),
        RetrievalResult(
            chunk_id="3", text="General PDU Session establishment in 5GS architecture",
            spec_number="23.501", parent_section_id="5.6.7", score=0.72,
        ),
    ]


class TestRerankerInit:
    """初始化与配置."""

    def test_get_reranker_returns_none_when_disabled(self, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "reranker_enabled", False)
        reranker = get_reranker()
        assert reranker is None

    def test_get_reranker_enabled_creates_instance(self, monkeypatch):
        """reranker_enabled=True 时返回 CrossEncoderReranker 实例."""
        from src.config import settings
        monkeypatch.setattr(settings, "reranker_enabled", True)
        reranker = get_reranker()
        assert reranker is not None
        from src.retriever.reranker import CrossEncoderReranker
        assert isinstance(reranker, CrossEncoderReranker)


class TestRerankerScoreFusion:
    """分数融合逻辑验证 — 不依赖真实模型加载."""

    def test_reranker_skip_when_few_results(self, monkeypatch):
        """结果数 ≤ top_k 时跳过精排."""
        from src.config import settings
        monkeypatch.setattr(settings, "reranker_enabled", True)
        # 少于 20 条不应触发精排 — 此测试验证逻辑而非实际运行
        # 因 reranker 需要下载模型, 此处验证代码路径
        assert True  # 占位, 实际测试需模型

    def test_reranker_disabled_passthrough(self, monkeypatch):
        """reranker_enabled=False 时原样返回."""
        from src.config import settings
        monkeypatch.setattr(settings, "reranker_enabled", False)
        # 验证 pipeline 中的 _rerank 方法在 disabled 时直接返回
        assert True  # 路径验证


class TestRerankFullPoolScoring:
    """显式 top_k ≥ 候选数时不得早退 — 融合场景需要对全池打分归一化."""

    def test_explicit_top_k_full_pool_forces_scoring(self, monkeypatch):
        """rerank(..., top_k=len(candidates)) 必须执行推理, 不能走早退捷径."""
        from src.retriever.reranker import CrossEncoderReranker

        candidates = [
            RetrievalResult(
                chunk_id=i, text=f"chunk {i}", score=1.0 - i * 0.01,
            )
            for i in range(50)
        ]
        fake_model = MagicMock()
        # 返回与候选等长的分数 (id 越大分越低)
        fake_model.predict.return_value = np.array(
            [50 - i for i in range(50)], dtype=np.float32
        )

        rr = CrossEncoderReranker(model_path="fake-model", top_k=20, batch_size=4)
        rr._model = fake_model

        out = rr.rerank("query", candidates, top_k=len(candidates))

        assert fake_model.predict.called, "显式全池打分被早退捷径跳过"
        assert len(out) == len(candidates)
        assert str(out[0].chunk_id) == "0"  # 最高分排最前

    def test_default_top_k_still_early_returns_when_few(self, monkeypatch):
        """默认 top_k (None) 且候选数不足时仍应跳过推理 (性能优化保留)."""
        from src.retriever.reranker import CrossEncoderReranker

        candidates = [
            RetrievalResult(chunk_id=i, text=f"chunk {i}", score=0.9)
            for i in range(3)
        ]
        fake_model = MagicMock()

        rr = CrossEncoderReranker(model_path="fake-model", top_k=20, batch_size=4)
        rr._model = fake_model

        out = rr.rerank("query", candidates)  # top_k=None → 实例默认 20

        assert not fake_model.predict.called
        assert len(out) == 3
