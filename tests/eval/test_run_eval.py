"""eval/run_eval.py 单元测试 — load_test_set / format_report / _dry_run."""

import json
import tempfile
from pathlib import Path

from tests.eval.metrics import EvalReport
from tests.eval.run_eval import format_report, load_test_set


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
        assert "3GPP RAG 检索评测报告" in output
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
