"""测试 Feedback — 反馈统计与报告生成.

真实 API: get_stats() + generate_report()，基于 SQLite feedback.db.
"""

import json
import sqlite3
import time
from pathlib import Path

import pytest


@pytest.fixture
def temp_feedback_db(tmp_path, monkeypatch):
    """创建临时 feedback.db 并 patch 路径."""
    db_path = tmp_path / "feedback.db"
    monkeypatch.setattr("src.generator.feedback._DB_PATH", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            sources     TEXT    NOT NULL DEFAULT '[]',
            rating      TEXT    NOT NULL CHECK(rating IN ('up', 'down')),
            comment     TEXT    DEFAULT '',
            created_at  REAL    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    yield db_path
    try:
        db_path.unlink(missing_ok=True)
    except OSError:
        pass


def _insert_feedback(db_path: Path, query: str, answer: str, rating: str, sources: list | None = None):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO feedback (query, answer, sources, rating, comment, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (query, answer, json.dumps(sources or []), rating, "", time.time()),
    )
    conn.commit()
    conn.close()


class TestGetStats:
    """get_stats() 统计聚合."""

    def test_empty_db(self, temp_feedback_db):
        from src.generator.feedback import get_stats
        stats = get_stats()
        assert stats["total"] == 0
        assert stats["up"] == 0
        assert stats["down"] == 0
        assert stats["up_ratio"] == 0.0

    def test_all_up(self, temp_feedback_db):
        for i in range(5):
            _insert_feedback(temp_feedback_db, f"q{i}", f"a{i}", "up")
        from src.generator.feedback import get_stats
        stats = get_stats()
        assert stats["total"] == 5
        assert stats["up"] == 5
        assert stats["down"] == 0
        assert stats["up_ratio"] == 1.0

    def test_mixed_ratings(self, temp_feedback_db):
        ratings = ["up", "up", "down", "up", "down", "up", "up"]
        for i, r in enumerate(ratings):
            _insert_feedback(temp_feedback_db, f"q{i}", f"a{i}", r)
        from src.generator.feedback import get_stats
        stats = get_stats()
        assert stats["total"] == 7
        assert stats["up"] == 5
        assert stats["down"] == 2
        assert 0.7 < stats["up_ratio"] < 0.72

    def test_db_not_exist(self, monkeypatch, tmp_path):
        """数据库文件不存在时返回零统计."""
        monkeypatch.setattr("src.generator.feedback._DB_PATH", tmp_path / "nonexistent.db")
        from src.generator.feedback import get_stats
        stats = get_stats()
        assert stats["total"] == 0
        assert stats["up_ratio"] == 0.0


class TestGenerateReport:
    """generate_report() Markdown 报告."""

    def test_empty_db_report(self, temp_feedback_db):
        from src.generator.feedback import generate_report
        report = generate_report()
        assert "暂无反馈数据" in report or "feedback.db" in report

    def test_report_with_data(self, temp_feedback_db):
        for i in range(3):
            _insert_feedback(temp_feedback_db, f"query {i}", f"answer {i}", "up")
        _insert_feedback(temp_feedback_db, "bad query", "bad answer", "down")
        from src.generator.feedback import generate_report
        report = generate_report()
        assert "总体统计" in report
        assert "4" in report
