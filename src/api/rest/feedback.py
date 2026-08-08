"""用户反馈 API — 👍👎 评分 + 可选评论，SQLite 持久化."""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.auth import require_admin
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Feedback"])

# ── 数据库路径 ──
_DB_PATH = settings.project_root / "data" / "feedback.db"


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接（自动建表）."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_rating
        ON feedback(rating)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_feedback_created
        ON feedback(created_at)
    """)
    conn.commit()
    return conn


# ── 请求/响应模型 ──

class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="原始查询")
    answer: str = Field(..., min_length=1, description="LLM 回答全文")
    sources: list[dict] = Field(default_factory=list, description="溯源来源列表")
    rating: str = Field(..., pattern=r"^(up|down)$", description="up=👍, down=👎")
    comment: str = Field(default="", max_length=500, description="可选意见")


class FeedbackItem(BaseModel):
    id: int
    query: str
    answer: str
    sources: list[dict]
    rating: str
    comment: str
    created_at: str  # ISO 格式


class FeedbackStats(BaseModel):
    total: int
    up: int
    down: int
    up_ratio: float


# ── 端点 ──

@router.post("/feedback", status_code=201)
async def submit_feedback(req: FeedbackRequest) -> dict:
    """提交 👍/👎 反馈."""
    try:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO feedback (query, answer, sources, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.query,
                req.answer,
                json.dumps(req.sources, ensure_ascii=False),
                req.rating,
                req.comment.strip(),
                time.time(),
            ),
        )
        conn.commit()
        feedback_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        logger.info(
            "反馈已记录 id=%s rating=%s query=%.60s", feedback_id, req.rating, req.query,
        )
        return {"id": feedback_id, "status": "ok"}
    except Exception as e:
        logger.error("反馈记录失败: %s", e)
        raise HTTPException(status_code=500, detail=f"反馈记录失败: {e}")


@router.get("/feedback", response_model=list[FeedbackItem])
async def list_feedback(
    rating: str | None = Query(None, pattern=r"^(up|down)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _admin=Depends(require_admin),
) -> list[FeedbackItem]:
    """查询反馈列表（管理后台专用, 需登录会话）."""
    conn = _get_conn()
    where = ""
    params: list = []
    if rating:
        where = "WHERE rating = ?"
        params.append(rating)
    rows = conn.execute(
        f"SELECT * FROM feedback {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()
    return [
        FeedbackItem(
            id=r["id"],
            query=r["query"],
            answer=r["answer"][:500],
            sources=json.loads(r["sources"]),
            rating=r["rating"],
            comment=r["comment"],
            created_at=_fmt_ts(r["created_at"]),
        )
        for r in rows
    ]


@router.get("/feedback/stats", response_model=FeedbackStats)
async def feedback_stats(_admin=Depends(require_admin)) -> FeedbackStats:
    """反馈统计（管理后台专用, 需登录会话）."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
    down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
    conn.close()
    return FeedbackStats(
        total=total,
        up=up,
        down=down,
        up_ratio=round(up / total, 4) if total > 0 else 0.0,
    )


def _fmt_ts(ts: float) -> str:
    """Unix timestamp → ISO 字符串."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
