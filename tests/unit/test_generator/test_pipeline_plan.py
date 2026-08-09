"""RAGPipeline.plan() 公开入口单元测试 — 委托 RetrievalPlanner.plan.

RED→GREEN: 当前 pipeline 仅暴露 search(), 无 plan() → 本测试失败;
新增公开 plan() 方法后通过.
"""

from unittest.mock import MagicMock

from src.generator.pipeline import RAGPipeline
from src.retriever.planner import RetrievalContext
from src.retriever.search import RetrievalResult


class TestPipelinePlanDelegation:

    def test_plan_delegates_to_planner(self):
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe._planner = MagicMock()
        ctx = RetrievalContext(
            query_lang="en",
            search_query="q",
            expanded_query="q",
            results=[RetrievalResult(chunk_id=1, text="t", score=0.5, spec_number="38.300")],
            initial_results=[RetrievalResult(chunk_id=2, text="t2", score=0.4, spec_number="38.211")],
        )
        pipe._planner.plan.return_value = ctx

        out = pipe.plan("q")

        assert out is ctx
        pipe._planner.plan.assert_called_once_with(
            "q", reranker_enabled=True, release=None, series=None, doc_type=None,
        )

    def test_plan_passes_filters_through(self):
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe._planner = MagicMock()
        pipe._planner.plan.return_value = RetrievalContext(
            query_lang="en", search_query="q", expanded_query="q",
        )

        pipe.plan("q", reranker_enabled=False, release="R18", series="38", doc_type="3gpp")

        pipe._planner.plan.assert_called_once_with(
            "q", reranker_enabled=False, release="R18", series="38", doc_type="3gpp",
        )
