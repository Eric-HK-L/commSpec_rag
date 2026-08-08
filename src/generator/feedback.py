"""用户反馈分析工具 — 统计报告生成 (与 api/rest/feedback.py 共享 SQLite DB).

CLI:
    python -m src.cli feedback report    生成反馈分析报告 (Markdown)
    python -m src.cli feedback stats     简要统计 (JSON)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections import Counter

from src.config import settings

logger = logging.getLogger(__name__)

_DB_PATH = settings.project_root / "data" / "feedback.db"


def _get_conn() -> sqlite3.Connection:
    """获取只读数据库连接."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_stats() -> dict:
    """反馈汇总统计."""
    if not _DB_PATH.exists():
        return {"total": 0, "up": 0, "down": 0, "up_ratio": 0.0}
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
        return {
            "total": total,
            "up": up,
            "down": down,
            "up_ratio": round(up / total, 4) if total > 0 else 0.0,
        }
    finally:
        conn.close()


def generate_report() -> str:
    """生成 Markdown 格式反馈分析报告.

    包含: 总体统计, 评分分布, 低分案例 Top-10, 差评关联规范 Top-10.
    """
    if not _DB_PATH.exists():
        return "暂无反馈数据 (feedback.db 不存在)。"

    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if total == 0:
            return "暂无反馈数据。"

        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
        up_ratio = round(up / total, 4) if total > 0 else 0.0

        lines = [
            "# 用户反馈分析报告",
            "",
            f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**数据来源:** `{_DB_PATH}`",
            "",
            "## 总体统计",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 总反馈数 | {total} |",
            f"| 👍 好评 | {up} ({up_ratio:.1%}) |",
            f"| 👎 差评 | {down} ({1 - up_ratio:.1%}) |",
            "",
        ]

        # 差评案例
        down_rows = conn.execute(
            "SELECT * FROM feedback WHERE rating='down' ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        if down_rows:
            lines.append("## 👎 差评案例 Top-10")
            lines.append("")
            lines.append("| # | 查询 | 时间 |")
            lines.append("|---|---|---|")
            for i, row in enumerate(down_rows, 1):
                ts = time.strftime("%m-%d %H:%M", time.localtime(row["created_at"]))
                q = row["query"][:80]
                lines.append(f"| {i} | {q} | {ts} |")

        # 差评关联的规范
        all_down = conn.execute(
            "SELECT sources FROM feedback WHERE rating='down'"
        ).fetchall()
        if all_down:
            spec_counter: Counter = Counter()
            for row in all_down:
                try:
                    sources = json.loads(row["sources"])
                    for s in sources:
                        sn = s.get("spec_number", "")
                        if sn:
                            spec_counter[sn] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
            if spec_counter:
                lines.append("")
                lines.append("## 差评关联规范 Top-10")
                lines.append("")
                lines.append("| 规范 | 差评次数 |")
                lines.append("|---|---|")
                for spec, count in spec_counter.most_common(10):
                    lines.append(f"| {spec} | {count} |")

        # 时间分布
        times = conn.execute(
            "SELECT created_at, rating FROM feedback ORDER BY created_at"
        ).fetchall()
        if len(times) >= 10:
            first_ts = times[0]["created_at"]
            last_ts = times[-1]["created_at"]
            days = max(1, (last_ts - first_ts) / 86400)
            lines.append("")
            lines.append("## 时间分布")
            lines.append("")
            lines.append(f"- 最早反馈: {time.strftime('%Y-%m-%d %H:%M', time.localtime(first_ts))}")
            lines.append(f"- 最新反馈: {time.strftime('%Y-%m-%d %H:%M', time.localtime(last_ts))}")
            lines.append(f"- 日均反馈: {total / days:.1f} 条/天")

        return "\n".join(lines)
    finally:
        conn.close()
