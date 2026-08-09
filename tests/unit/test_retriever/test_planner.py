"""planner.py 单元测试 — 查询扩展缓存/跳过规则/批量嵌入."""

import threading

import numpy as np
from cachetools import TTLCache

from src.retriever.planner import (
    RetrievalPlanner,
    _history_fingerprint,
    _is_precise_query,
)


def _make_planner() -> RetrievalPlanner:
    """轻量构造 — 绕过完整初始化 (避免加载 xref graph/模型)."""
    planner = object.__new__(RetrievalPlanner)
    planner._expand_cache = TTLCache(maxsize=16, ttl=3600)
    planner._expand_lock = threading.Lock()
    planner._llm = None
    return planner


class TestIsPreciseQuery:

    def test_spec_and_section_skip(self):
        assert _is_precise_query("TS 38.413 §8.3.1 PDU Session Setup procedure")
        assert _is_precise_query("38.413 section 8.3.1 PDU setup")
        assert _is_precise_query("38.413 §8.3.1")

    def test_spec_only_not_skip(self):
        assert not _is_precise_query("TS 38.413 PDU session setup")

    def test_general_query_not_skip(self):
        assert not _is_precise_query("What is the PDU session establishment procedure?")
        assert not _is_precise_query("PDU session")

    def test_section_only_not_skip(self):
        assert not _is_precise_query("§8.3.1 的内容是什么")


class TestHistoryFingerprint:

    def test_empty(self):
        assert _history_fingerprint(None) == ""
        assert _history_fingerprint([]) == ""

    def test_includes_roles_and_content(self):
        h = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
        fp = _history_fingerprint(h)
        assert "user" in fp and "你好" in fp

    def test_different_history_differs(self):
        h1 = [{"role": "user", "content": "a"}]
        h2 = [{"role": "user", "content": "b"}]
        assert _history_fingerprint(h1) != _history_fingerprint(h2)


class TestExpandQuery:

    def test_cache_hits_on_second_call(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        planner._llm.chat.return_value = "PDU session setup NGAP N2 procedure"

        r1 = planner._expand_query("pdu session setup")
        r2 = planner._expand_query("PDU Session Setup")  # 大小写归一化后命中缓存
        assert r1 == r2
        assert planner._llm.chat.call_count == 1

    def test_precise_query_skips_llm(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        planner._llm.chat.return_value = "should not be used"

        out = planner._expand_query("TS 38.413 §8.3.1 PDU Session Setup")
        assert out == "TS 38.413 §8.3.1 PDU Session Setup"
        planner._llm.chat.assert_not_called()

    def test_disabled_returns_query_as_is(self, monkeypatch):
        from unittest.mock import MagicMock

        from src.config import settings
        monkeypatch.setattr(settings, "query_expansion_enabled", False)

        planner = _make_planner()
        planner._llm = MagicMock()
        out = planner._expand_query("pdu session")
        assert out == "pdu session"
        planner._llm.chat.assert_not_called()

    def test_llm_failure_falls_back_to_query(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        planner._llm.chat.side_effect = RuntimeError("API down")

        out = planner._expand_query("pdu session")
        assert out == "pdu session"


class TestGetQueryEmbeddings:

    def test_batch_embed_single_call(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        planner._llm.embed.return_value = [[0.1] * 1024, [0.2] * 1024]

        embeds = planner._get_query_embeddings(["a", "b"])
        assert len(embeds) == 2
        assert embeds[0].shape == (1024,)
        assert embeds[0].dtype == np.float32
        planner._llm.embed.assert_called_once_with(["a", "b"])

    def test_empty_input(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        assert planner._get_query_embeddings([]) == []
        planner._llm.embed.assert_not_called()

    def test_failure_falls_back_to_zero_vectors(self):
        from unittest.mock import MagicMock

        planner = _make_planner()
        planner._llm = MagicMock()
        planner._llm.embed.side_effect = RuntimeError("embed failed")

        embeds = planner._get_query_embeddings(["a", "b"])
        assert len(embeds) == 2
        assert embeds[0].shape == (1024,)
        assert np.all(embeds[0] == 0)


class TestRerankFusionScope:
    """_rerank 融合作用域 — 必须覆盖全候选池, 而非 reranker 截断后的 top_k.

    缺陷: 旧实现先 reranker.rerank(top_k) 截断到 20, 再只在 20 条内融合 —
    RRF 排 3-5 但 reranker 排 21+ 的真阳性直接出局.
    修复: 对全候选池打分并融合, 最后才截断到 top_k.
    """

    def test_rrf_top5_but_reranker_rank21_survives(self, monkeypatch):
        """pool=50 / top_k=20 — RRF 排 5 但 reranker 排 21 的 chunk 必须进入最终 top-20."""
        from src.config import settings
        from src.retriever.search import RetrievalResult

        monkeypatch.setattr(settings, "reranker_enabled", True)

        top_k = 20
        # 候选池 50 条: 原始分 (RRF 序) 降序 1.0 → 0.51
        results = [
            RetrievalResult(
                chunk_id=i, text=f"chunk {i}", score=1.0 - i * 0.01,
                spec_number="38.413", parent_section_id="8.3.1",
            )
            for i in range(50)
        ]
        # 真阳性: RRF 排 5 (原始分 0.965), 但 reranker 分把它压到第 21 名
        target = RetrievalResult(
            chunk_id="t", text="target true positive", score=0.965,
            spec_number="38.413", parent_section_id="8.3.1",
        )
        results.insert(4, target)

        # Fake reranker: 分数 = -chunk_id (id 越大越差), target 固定 -20.5 → 排名 21
        class _FakeReranker:
            def rerank(self, query, candidates, top_k=None):
                k = top_k if top_k is not None else len(candidates)
                for c in candidates:
                    c.score = float(
                        -20.5 if str(c.chunk_id) == "t" else -int(c.chunk_id)
                    )
                scored = sorted(candidates, key=lambda c: c.score, reverse=True)
                return scored[:k]

        monkeypatch.setattr("src.retriever.planner.get_reranker", lambda: _FakeReranker())

        planner = _make_planner()
        out = planner._rerank("query", results, top_k)

        ids = [str(r.chunk_id) for r in out]
        assert len(out) <= top_k
        assert "t" in ids, (
            "RRF 排 5 但 reranker 排 21+ 的真阳性被融合作用域截断丢弃"
        )

    def test_fused_scores_are_python_float(self, monkeypatch):
        """融合后分数必须是 JSON 可序列化的 Python float (SSE sources 依赖)."""
        from src.config import settings
        from src.retriever.search import RetrievalResult

        monkeypatch.setattr(settings, "reranker_enabled", True)

        top_k = 20
        results = [
            RetrievalResult(
                chunk_id=i, text=f"chunk {i}", score=1.0 - i * 0.01,
            )
            for i in range(25)
        ]

        class _FakeReranker:
            def rerank(self, query, candidates, top_k=None):
                k = top_k if top_k is not None else len(candidates)
                for c in candidates:
                    c.score = float(-int(c.chunk_id))
                scored = sorted(candidates, key=lambda c: c.score, reverse=True)
                return scored[:k]

        monkeypatch.setattr("src.retriever.planner.get_reranker", lambda: _FakeReranker())

        planner = _make_planner()
        out = planner._rerank("query", results, top_k)

        import json
        payload = [{"chunk_id": str(r.chunk_id), "score": r.score} for r in out]
        json.dumps(payload)  # 不抛 TypeError = 分数可序列化
        assert all(isinstance(r.score, float) for r in out)
