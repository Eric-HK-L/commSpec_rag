"""批量嵌入生成器 — API 优先 + 本地模型降级 + 双层缓存.

缓存策略:
  1. SQLite 缓存 (embedding_cache.py) — 批量查询, O(log n)
  2. 文件缓存 (.npy) — 向后兼容, 逐文件读写
  两者可并存: SQLite 优先, 文件缓存兜底.

嵌入策略:
  1. 云端 API (OpenAI 兼容) → 失败降级
  2. 本地 sentence-transformers → 兜底
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import settings
from src.ingestion.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """批量文本嵌入生成, 含速率控制和缓存."""

    def __init__(
        self,
        batch_size: int = 32,
        cache_dir: str | None = None,
        sqlite_cache: EmbeddingCache | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        self._batch_size = batch_size
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._sqlite_cache = sqlite_cache
        self._on_progress = on_progress
        self._llm_client = None

    @property
    def _client(self):
        """懒加载 LLM 客户端."""
        if self._llm_client is None:
            from src.generator.llm_client import LLMClient
            self._llm_client = LLMClient()
        return self._llm_client

    # ── 主接口 ──

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """对批量文本生成嵌入.

        Args:
            texts: 文本列表.

        Returns:
            shape=(len(texts), embedding_dim) 的 numpy 数组.
        """
        if not texts:
            return np.empty((0, settings.embedding_dimension), dtype=np.float32)

        total = len(texts)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, total, self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embed_with_fallback(batch)
            all_embeddings.extend(embeddings)

            if self._on_progress:
                self._on_progress(min(i + self._batch_size, total), total)

            # 速率控制 — 仅 API 模式需要，本地 BGE 无需限速
            if settings.embedding_provider == "api":
                time.sleep(0.3)

        result = np.array(all_embeddings, dtype=np.float32)
        return result

    def embed_single(self, text: str) -> np.ndarray:
        """单文本嵌入."""
        return self.embed_batch([text])[0]

    # ── 嵌入逻辑 ──

    def _embed_with_fallback(self, texts: list[str]) -> list[np.ndarray]:
        """API 优先 → 本地降级 → 零向量兜底."""
        # 尝试缓存
        cached, uncached_indices = self._check_cache(texts)
        if not uncached_indices:
            return cached

        uncached_texts = [texts[i] for i in uncached_indices]

        # 1. 云端 API
        try:
            api_embeddings = self._client.embed(uncached_texts)
            self._update_cache(uncached_texts, api_embeddings)
            return self._merge_cached(cached, api_embeddings, uncached_indices, len(texts))
        except Exception as e:
            logger.debug("API 嵌入失败, 降级本地: %s", e)

        # 2. 本地模型 (由 llm_client.embed 内部处理降级)
        try:
            local_embeddings = self._client.embed(uncached_texts)
            self._update_cache(uncached_texts, local_embeddings)
            return self._merge_cached(cached, local_embeddings, uncached_indices, len(texts))
        except Exception as e:
            logger.error("嵌入完全失败: %s", e)
            raise

    # ── 缓存 ──

    def _check_cache(self, texts: list[str]) -> tuple[list[np.ndarray], list[int]]:
        """检查双层缓存, 返回 (已缓存的嵌入, 未缓存的索引列表).

        优先级: SQLite → 文件缓存 → 未命中.
        """
        n = len(texts)
        result: dict[int, np.ndarray] = {}  # index → embedding
        uncached_indices: set[int] = set(range(n))

        # 第1层: SQLite 批量查询
        if self._sqlite_cache:
            sqlite_hits = self._sqlite_cache.get_batch(texts)
            for idx in list(uncached_indices):
                key = self._sqlite_cache.text_key(texts[idx])
                if key in sqlite_hits:
                    result[idx] = sqlite_hits[key]
                    uncached_indices.discard(idx)

        # 第2层: 文件缓存 (npy), 跳过已命中
        if self._cache_dir:
            for idx in sorted(uncached_indices):
                cache_path = self._cache_path(texts[idx])
                if cache_path and cache_path.exists():
                    result[idx] = np.load(cache_path)
                    uncached_indices.discard(idx)
                    # 回填 SQLite
                    if self._sqlite_cache:
                        try:
                            self._sqlite_cache.put(texts[idx], result[idx])
                        except Exception:
                            pass

        # 按原始顺序排列缓存结果
        cached_list = [result[i] for i in sorted(result)]
        uncached_list = sorted(uncached_indices)
        return cached_list, uncached_list

    def _update_cache(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """将新嵌入写入双层缓存 (SQLite + 文件)."""
        emb_arr = np.array(embeddings, dtype=np.float32)

        # SQLite 批量写入
        if self._sqlite_cache:
            try:
                self._sqlite_cache.put_batch(texts, emb_arr)
            except Exception as e:
                logger.debug("SQLite 缓存写入失败: %s", e)

        # 文件缓存 (npy)
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            for text, emb in zip(texts, embeddings):
                cache_path = self._cache_path(text)
                if cache_path:
                    np.save(cache_path, np.array(emb, dtype=np.float32))

    def _cache_path(self, text: str) -> Path | None:
        """计算文本的缓存路径."""
        if not self._cache_dir:
            return None
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{h}.npy"

    @staticmethod
    def _merge_cached(
        cached: list[np.ndarray],
        fresh: list[list[float]],
        fresh_indices: list[int],
        total: int,
    ) -> list[np.ndarray]:
        """合并缓存和新嵌入, 按原始顺序排列."""
        result: list[np.ndarray | None] = [None] * total
        fresh_arr = [np.array(e, dtype=np.float32) for e in fresh]

        # 填入缓存
        fresh_idx = 0
        cached_idx = 0
        for i in range(total):
            if fresh_idx < len(fresh_indices) and fresh_indices[fresh_idx] == i:
                result[i] = fresh_arr[fresh_idx]
                fresh_idx += 1
            else:
                result[i] = cached[cached_idx]
                cached_idx += 1

        return [r for r in result if r is not None]  # type: ignore[misc]
