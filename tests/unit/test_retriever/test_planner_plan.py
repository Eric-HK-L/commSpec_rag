"""planner.plan() 全链路单元测试 — 暴露精排前候选池 (initial_results).

RED→GREEN: 当前 RetrievalContext 无 initial_results 字段 → 本测试失败;
在 plan() 的 _post_process_results (精排) 之前捕获候选池后通过.
"""

import threading

import numpy as np
from cachetools import TTLCache

from src.retriever.planner import RetrievalPlanner


class TestPlanExposesInitialResults:

    def test_initial_results_captured_before_rerank(self, monkeypatch):
        from unittest.mock import MagicMock

        from src.retriever import planner as planner_mod
        from src.retriever.search import RetrievalResult

        p = object.__new__(RetrievalPlanner)
        p._retriever = MagicMock()
        p._multi_hop = MagicMock()
        p._online = MagicMock()
        p._graph_expander = None
        p._expand_cache = TTLCache(maxsize=16, ttl=3600)
        p._expand_lock = threading.Lock()

        candidates = [
            RetrievalResult(chunk_id=1, text="5QI table", score=0.9, spec_number="38.300"),
            RetrievalResult(chunk_id=2, text="QoS flow", score=0.8, spec_number="38.211"),
        ]
        p._retriever.search.return_value = candidates
        p._expand_query = MagicMock(return_value="expanded q")
        p._get_query_embedding = MagicMock(return_value=np.zeros(4, dtype=np.float32))

        seen: dict = {}

        def fake_post_process(query, expanded_query, query_embedding, results, top_k,
                              reranker_enabled=True, filter_expr=None):
            seen["pre_rerank"] = results
            return list(results)

        p._post_process_results = fake_post_process

        monkeypatch.setattr(planner_mod, "detect_language", lambda q: "en")
        monkeypatch.setattr(planner_mod, "is_taxonomy_query", lambda q: False)
        monkeypatch.setattr(planner_mod, "filter_low_quality", lambda r, k: r)
        monkeypatch.setattr(planner_mod, "needs_multi_hop", lambda r: False)
        monkeypatch.setattr(planner_mod, "filter_noise", lambda r: r)
        monkeypatch.setattr(
            planner_mod, "evaluate_quality",
            lambda r: MagicMock(overall_ok=True, density=1.0, diversity=1.0, coverage=2),
        )
        monkeypatch.setattr(
            planner_mod, "diagnose_quality",
            lambda q, n: MagicMock(reason="", should_rewrite=False, should_expand=False, should_suggest=False),
        )
        release_intent = MagicMock()
        release_intent.type.value = "none"
        monkeypatch.setattr(planner_mod, "detect_release_intent", lambda q: release_intent)
        monkeypatch.setattr(planner_mod, "record_search", lambda *a, **k: None)
        monkeypatch.setattr(planner_mod, "settings", MagicMock(enable_online_search=False))

        ctx = p.plan("What is 5QI?")

        assert ctx.initial_results == candidates
        assert seen["pre_rerank"] == candidates
        assert ctx.results == candidates
