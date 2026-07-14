"""嵌入缓存 — SQLite 持久化, SHA256 索引, 避免重复计算嵌入向量.

与 BatchEmbedder 文件缓存的区别:
  - 单文件 SQLite 替代散文件 .npy → 减少 I/O 系统调用
  - 批量查询: SELECT ... WHERE hash IN (...) → O(log n) vs O(n)
  - 自动去重 + 大小统计
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


def _default_db_path() -> Path:
    """嵌入缓存 SQLite 文件路径 (从 settings 读取)."""
    p = settings.embedding_cache_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class EmbeddingCache:
    """SQLite 嵌入缓存.

    用法:
        cache = EmbeddingCache()
        emb = cache.get("some text")        # → np.ndarray or None
        cache.put("some text", emb)         # 持久化
        hits = cache.get_batch(texts)       # 批量查询 → {hash: np.ndarray}
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._dim = settings.embedding_dimension
        self._init_db()

    # ── 数据库初始化 ──

    def _init_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                hash       TEXT    PRIMARY KEY,
                text_hash  TEXT    NOT NULL,
                embedding  BLOB    NOT NULL,
                dim        INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                hit_count  INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash
            ON embeddings(text_hash)
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ── 文本哈希 ──

    @staticmethod
    def text_key(text: str) -> str:
        """文本的 SHA256 标识."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ── 单条操作 ──

    def get(self, text: str) -> np.ndarray | None:
        """查询单条嵌入."""
        key = self.text_key(text)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding, dim FROM embeddings WHERE hash = ?", (key,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE embeddings SET hit_count = hit_count + 1 WHERE hash = ?",
                (key,),
            )
            conn.commit()
            conn.close()
            arr = np.frombuffer(row[0], dtype=np.float32)
            if len(arr) == row[1]:
                return arr.copy()
            logger.warning("缓存维度不匹配: 期望 %d, 实际 %d", row[1], len(arr))
        conn.close()
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """存储单条嵌入."""
        key = self.text_key(text)
        text_hash = key[:16]
        arr = np.asarray(embedding, dtype=np.float32)
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO embeddings
               (hash, text_hash, embedding, dim, created_at, hit_count)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (key, text_hash, arr.tobytes(), len(arr), time.time()),
        )
        conn.commit()
        conn.close()

    # ── 批量操作 ──

    def get_batch(self, texts: list[str]) -> dict[str, np.ndarray]:
        """批量查询: 返回 {text_key: embedding}.

        Args:
            texts: 文本列表.

        Returns:
            仅包含已缓存条目的字典. 未命中不在字典中.
        """
        if not texts:
            return {}

        keys = [self.text_key(t) for t in texts]
        dict(zip(keys, texts))

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in keys)
        rows = conn.execute(
            f"SELECT hash, embedding, dim FROM embeddings WHERE hash IN ({placeholders})",
            keys,
        ).fetchall()

        # 更新命中计数
        if rows:
            hit_keys = [r[0] for r in rows]
            conn.executemany(
                "UPDATE embeddings SET hit_count = hit_count + 1 WHERE hash = ?",
                [(k,) for k in hit_keys],
            )
            conn.commit()
        conn.close()

        result: dict[str, np.ndarray] = {}
        for row in rows:
            arr = np.frombuffer(row[1], dtype=np.float32)
            if len(arr) == row[2]:
                result[row[0]] = arr.copy()
        return result

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> int:
        """批量存储. 返回新增数量."""
        if len(texts) != len(embeddings):
            raise ValueError(f"texts ({len(texts)}) 与 embeddings ({len(embeddings)}) 数量不匹配")

        now = time.time()
        records: list[tuple] = []
        for text, emb in zip(texts, embeddings):
            key = self.text_key(text)
            arr = np.asarray(emb, dtype=np.float32)
            records.append((key, key[:16], arr.tobytes(), len(arr), now))

        conn = self._get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO embeddings
               (hash, text_hash, embedding, dim, created_at, hit_count)
               VALUES (?, ?, ?, ?, ?, 1)""",
            records,
        )
        conn.commit()
        conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        return len(records)

    # ── 统计 ──

    def stats(self) -> dict:
        """缓存统计."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        size_row = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(embedding)), 0) FROM embeddings"
        ).fetchone()
        conn.close()
        return {
            "total_entries": total,
            "size_bytes": size_row[0] if size_row else 0,
            "size_mb": round((size_row[0] if size_row else 0) / 1024 / 1024, 2),
            "db_path": str(self._db_path),
        }

    def clear(self) -> int:
        """清空缓存, 返回删除数量."""
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        logger.info("嵌入缓存已清空: %d 条", count)
        return count
