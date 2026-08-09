"""eval/run_eval.py 单元测试 — load_test_set / format_report / _dry_run / plan() 全链路."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tests.eval.metrics import EvalReport
from tests.eval.run_eval import format_report, load_test_set, run_eval


class TestLoadTestSet:
    """load_test_set — 从 JSON 加载测试集."""

    def test_load_valid(self):
        data = [{
            "question": "What is 5QI?",
            "expected_specs": ["23.501"],
            "expected_sections": ["5.7.4"],
            "difficulty": "easy",
            "multi_hop": False,
        }]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            samples = load_test_set(tmp)
            assert len(samples) == 1
            assert samples[0].question == "What is 5QI?"
            assert samples[0].expected_specs == ["23.501"]
            assert samples[0].difficulty == "easy"
            assert samples[0].multi_hop is False
        finally:
            Path(tmp).unlink()

    def test_load_with_defaults(self):
        data = [{"question": "test", "expected_specs": [], "expected_sections": []}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            samples = load_test_set(tmp)
            assert samples[0].difficulty == "medium"  # 默认值
            assert samples[0].multi_hop is False  # 默认值
        finally:
            Path(tmp).unlink()

    def test_load_multiple(self):
        data = [
            {"question": "q1", "expected_specs": ["a"], "expected_sections": []},
            {"question": "q2", "expected_specs": ["b"], "expected_sections": []},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            samples = load_test_set(tmp)
            assert len(samples) == 2
        finally:
            Path(tmp).unlink()

    def test_load_real_test_set(self):
        """验证真实 test_set.json 可以正确加载."""
        test_set_path = Path(__file__).resolve().parent / "test_set.json"
        if test_set_path.exists():
            samples = load_test_set(str(test_set_path))
            assert len(samples) > 0
            for s in samples:
                assert s.question
                assert isinstance(s.expected_specs, list)


class TestFormatReport:
    """format_report — 评测报告 Markdown 格式化."""

    def test_basic_report(self):
        report = EvalReport(
            total=10,
            recall_at_5=0.85,
            recall_at_10=0.90,
            recall_at_20=0.95,
            mrr=0.72,
            ndcg_at_10=0.78,
        )
        output = format_report(report, 1500.0)
        assert "CommSpec RAG 检索评测报告" in output
        assert "**样本数**: 10" in output
        assert "0.85" in output
        assert "0.90" in output
        assert "0.95" in output
        assert "0.72" in output
        assert "0.78" in output

    def test_report_with_difficulty(self):
        report = EvalReport(
            total=5,
            recall_at_5=0.8, recall_at_10=0.85, recall_at_20=0.9,
            mrr=0.7, ndcg_at_10=0.75,
            by_difficulty={
                "easy": {"count": 2, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
                "hard": {"count": 3, "recall@5": 0.5, "recall@10": 0.6, "mrr": 0.4, "ndcg@10": 0.45},
            },
        )
        output = format_report(report, 1000.0)
        assert "easy" in output
        assert "hard" in output

    def test_report_with_multi_hop(self):
        report = EvalReport(
            total=5,
            recall_at_5=0.8, recall_at_10=0.85, recall_at_20=0.9,
            mrr=0.7, ndcg_at_10=0.75,
            by_multi_hop={
                True: {"count": 2, "recall@5": 0.6, "recall@10": 0.7, "mrr": 0.5, "ndcg@10": 0.55},
                False: {"count": 3, "recall@5": 0.9, "recall@10": 0.95, "mrr": 0.85, "ndcg@10": 0.9},
            },
        )
        output = format_report(report, 1000.0)
        assert "多跳检索" in output
        assert "单步检索" in output

    def test_elapsed_formatting(self):
        report = EvalReport(
            total=1, recall_at_5=1.0, recall_at_10=1.0, recall_at_20=1.0,
            mrr=1.0, ndcg_at_10=1.0,
        )
        output = format_report(report, 2500.0)
        assert "2.5s" in output  # 2500ms → 2.5s


class TestRunEvalUsesPlanPath:
    """run_eval 必须走 plan() 全链路 — 不能回退到轻量 search() 路径.

    RED→GREEN: 当前实现调用 pipeline.search() → 本测试失败 (plan 未被调用);
    改走 pipeline.plan() 后通过. 该断言保障候选池=reranker_top_k 且
    多跳/图扩展/taxonomy 分解/filter_noise 全部参与评测, 而非 20 候选无重排.
    """

    def test_eval_calls_plan_not_search(self, tmp_path, monkeypatch):
        from src.retriever.search import RetrievalResult

        class FakeStore:
            """假 MilvusStore — 仅满足 run_eval 初始化路径."""
            count = 100

            def __init__(self, **kwargs):
                pass

            def connect(self):
                pass

        class FakePipeline:
            """假 RAGPipeline — 仅实现 plan(); search() 被调用即失败."""
            plan_calls: list[str] = []

            def __init__(self, vector_store=None, llm_client=None):
                pass

            def plan(self, query, reranker_enabled=True, **kwargs):
                FakePipeline.plan_calls.append(query)
                return SimpleNamespace(
                    results=[
                        RetrievalResult(chunk_id=1, text="t1", score=0.9, spec_number="38.300"),
                        RetrievalResult(chunk_id=2, text="t2", score=0.8, spec_number="38.211"),
                    ],
                    initial_results=[
                        RetrievalResult(chunk_id=3, text="t3", score=0.7, spec_number="38.211"),
                        RetrievalResult(chunk_id=4, text="t4", score=0.6, spec_number="38.300"),
                    ],
                )

            def search(self, *args, **kwargs):
                raise AssertionError("评测必须走 plan() 全链路, 不得调用轻量 search()")

        monkeypatch.setattr("src.retriever.milvus_store.MilvusStore", FakeStore)
        monkeypatch.setattr("src.generator.pipeline.RAGPipeline", FakePipeline)
        FakePipeline.plan_calls = []

        test_set = tmp_path / "test_set.json"
        test_set.write_text(json.dumps([{
            "question": "What is 5QI?",
            "expected_specs": ["38.300"],
            "expected_sections": [],
        }]), encoding="utf-8")

        run_eval(str(test_set), checkpoint=str(tmp_path / "ckpt.json"))

        assert FakePipeline.plan_calls == ["What is 5QI?"]

    def test_eval_reports_initial_and_final_recall(self, tmp_path, monkeypatch):
        """重排前 (initial) 与重排后 (final) 两条召回都要进报告."""
        from src.retriever.search import RetrievalResult

        class FakeStore:
            count = 100

            def __init__(self, **kwargs):
                pass

            def connect(self):
                pass

        class FakePipeline:
            def __init__(self, vector_store=None, llm_client=None):
                pass

            def plan(self, query, reranker_enabled=True, **kwargs):
                return SimpleNamespace(
                    # 初检: 38.300 排第 2 (Recall@5=1.0); 重排后: 38.300 排第 1
                    initial_results=[
                        RetrievalResult(chunk_id=1, text="t1", score=0.7, spec_number="38.211"),
                        RetrievalResult(chunk_id=2, text="t2", score=0.6, spec_number="38.300"),
                    ],
                    results=[
                        RetrievalResult(chunk_id=3, text="t3", score=0.9, spec_number="38.300"),
                        RetrievalResult(chunk_id=4, text="t4", score=0.8, spec_number="38.211"),
                    ],
                )

        monkeypatch.setattr("src.retriever.milvus_store.MilvusStore", FakeStore)
        monkeypatch.setattr("src.generator.pipeline.RAGPipeline", FakePipeline)

        test_set = tmp_path / "test_set.json"
        test_set.write_text(json.dumps([{
            "question": "5QI 定义",
            "expected_specs": ["38.300"],
            "expected_sections": [],
        }]), encoding="utf-8")

        run_eval(str(test_set), checkpoint=str(tmp_path / "ckpt.json"))

        report_md = (tmp_path / "eval_report.md").read_text(encoding="utf-8")
        assert "初检" in report_md
        assert "重排后" in report_md
        assert "search" in report_md

    def test_checkpoint_skips_computed_samples(self, tmp_path, monkeypatch):
        """checkpoint 缓存: 已算样本重跑时跳过, 不再调用 plan()."""
        from src.retriever.search import RetrievalResult

        class FakeStore:
            count = 100

            def __init__(self, **kwargs):
                pass

            def connect(self):
                pass

        class FakePipeline:
            plan_calls: list[str] = []

            def __init__(self, vector_store=None, llm_client=None):
                pass

            def plan(self, query, reranker_enabled=True, **kwargs):
                FakePipeline.plan_calls.append(query)
                return SimpleNamespace(
                    results=[RetrievalResult(chunk_id=1, text="t1", score=0.9, spec_number="38.300")],
                    initial_results=[],
                )

        monkeypatch.setattr("src.retriever.milvus_store.MilvusStore", FakeStore)
        monkeypatch.setattr("src.generator.pipeline.RAGPipeline", FakePipeline)

        test_set = tmp_path / "test_set.json"
        test_set.write_text(json.dumps([
            {"question": "q1", "expected_specs": ["38.300"], "expected_sections": []},
            {"question": "q2", "expected_specs": ["38.211"], "expected_sections": []},
        ]), encoding="utf-8")
        ckpt = str(tmp_path / "ckpt.json")

        FakePipeline.plan_calls = []
        run_eval(str(test_set), checkpoint=ckpt)
        assert FakePipeline.plan_calls == ["q1", "q2"]

        FakePipeline.plan_calls = []
        run_eval(str(test_set), checkpoint=ckpt)
        assert FakePipeline.plan_calls == []

        ckpt_data = json.loads(Path(ckpt).read_text(encoding="utf-8"))
        assert len(ckpt_data["samples"]) == 2
        for entry in ckpt_data["samples"].values():
            assert entry["retrieved_specs"] == ["38.300"]
