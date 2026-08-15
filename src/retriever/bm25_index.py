"""独立 BM25 稀疏检索索引 — 基于 rank-bm25 (Okapi BM25).

绕过 Milvus 原生 BM25 Function 的 pymilvus 2.x 不兼容问题，
在 Python 侧构建 BM25 索引 + RRF 融合。
"""

from __future__ import annotations

import logging
import pickle
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


class BM25Indexer:
    """Standalone BM25 索引器。

    与 Milvus Dense 检索配合，通过 (doc_id, spec_number, chunk_index) 三元组
    作为跨库共享文档标识符进行 RRF 融合。
    """

    def __init__(self, index_path: str | Path | None = None):
        self._index_path = Path(index_path) if index_path else settings.bm25_index_path
        self._bm25: Optional[BM25Okapi] = None
        # doc_key -> 在 BM25 语料中的位置索引
        self._doc_keys: list[str] = []
        # 每 chunk 的元数据 (release/series/doc_type), 与 _doc_keys 等长
        self._meta: list[dict] = []
        # 原始文本（用于调试/显示）
        self._texts: list[str] = []
        # BM25 索引非线程安全 — build/load/重建 与并发检索需互斥
        self._lock = threading.RLock()

    # ── 索引构建 ──

    def build(
        self,
        texts: list[str],
        doc_ids: list[str],
        spec_numbers: list[str],
        chunk_indices: list[int],
        metadata: list[dict] | None = None,
    ) -> None:
        """从语料构建 BM25 索引。

        Args:
            texts: 所有 chunk 文本列表。
            doc_ids: 对应的文档 ID 列表。
            spec_numbers: 对应的规范号列表。
            chunk_indices: 对应的 chunk 序号列表。
            metadata: 与 texts 等长的每 chunk 元数据 (release/series/doc_type 等),
                用于带过滤条件的稀疏检索。
        """
        n = len(texts)
        assert len(doc_ids) == len(spec_numbers) == len(chunk_indices) == n, \
            f"输入列表长度不一致: texts={n}, doc_ids={len(doc_ids)}, ..."
        if metadata is not None:
            assert len(metadata) == n, f"metadata 长度不一致: {len(metadata)} != {n}"

        logger.info("开始构建 BM25 索引, 语料量: %d 条", n)

        with self._lock:
            # 生成文档唯一键
            self._doc_keys = [
                f"{doc_ids[i]}|{spec_numbers[i]}|{chunk_indices[i]}"
                for i in range(n)
            ]
            self._meta = list(metadata) if metadata is not None else [{} for _ in range(n)]
            self._texts = texts

            # 分词
            tokenized = [self._tokenize(t) for t in texts]

            # 构建 BM25Okapi 索引
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 索引构建完成 (%d 条, 词表约 %d 词)",
                         n, len(self._bm25.idf) if self._bm25 else 0)

    # 域名感知分词: 规范号(38.211)保持整 token, 连字符/标点按非字母数字切分, 小写化.
    # 相比纯 split(): "PRACH-preamble" → ["prach", "preamble"], 使 "PRACH preamble" 类查询可命中;
    # "(38.211)" → ["38.211"], 保留规范号整体而非拆成 "38" "211".
    _TOKEN_RE = re.compile(r"\d{2}\.\d{3}|[a-z0-9]+")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """域名感知分词 — 规范号整 token + 连字符技术词拆分 + 小写化."""
        return BM25Indexer._TOKEN_RE.findall(text.lower())

    # ── 检索 ──

    def search(
        self, query: str, top_k: int = 100
    ) -> list[tuple[str, float, str]]:
        """BM25 检索。

        Args:
            query: 查询文本。
            top_k: 返回结果数。

        Returns:
            [(doc_key, bm25_score, text), ...] 列表。
        """
        with self._lock:
            if self._bm25 is None or not self._doc_keys:
                return []

            tokenized = self._tokenize(query)
            if not tokenized:
                return []

            scores = self._bm25.get_scores(tokenized)
            if len(scores) == 0:
                return []

            # 获取 top-k 索引
            limit = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:limit]

            results: list[tuple[str, float, str]] = []
            for idx in top_indices:
                score = float(scores[idx])
                if score <= 0:
                    continue
                results.append((
                    self._doc_keys[idx],
                    score,
                    self._texts[idx] if idx < len(self._texts) else "",
                ))

            return results

    def search_with_meta(
        self, query: str, top_k: int = 100
    ) -> list[tuple[str, float, str, dict]]:
        """BM25 检索并附带每条的元数据 (release/series/doc_type).

        Returns:
            [(doc_key, bm25_score, text, meta), ...] 列表.
        """
        results = self.search(query, top_k)
        # _meta 与 _doc_keys 等长且按下标对应; 检索结果是按分数排序的,
        # 必须用 doc_key 反查元数据, 否则过滤 (release/series/doc_type) 会错位
        with self._lock:
            key_to_idx = {key: i for i, key in enumerate(self._doc_keys)}
        return [
            (doc_key, score, text, self._meta[key_to_idx[doc_key]] if doc_key in key_to_idx else {})
            for doc_key, score, text in results
        ]

    # ── 持久化 ──

    def save(self) -> None:
        """保存 BM25 索引到磁盘 (pickle)。"""
        with self._lock:
            if self._bm25 is None:
                return
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "bm25": self._bm25,
                "doc_keys": self._doc_keys,
                "texts": self._texts,
                "meta": self._meta,
            }
            with open(self._index_path, "wb") as f:
                pickle.dump(data, f)
            logger.info("BM25 索引已保存: %s (%.1f MB)",
                         self._index_path,
                         self._index_path.stat().st_size / (1024 * 1024))

    def load(self) -> bool:
        """从磁盘加载 BM25 索引。

        Returns:
            True 如果加载成功。
        """
        with self._lock:
            if not self._index_path.exists():
                return False
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._doc_keys = data["doc_keys"]
            self._texts = data.get("texts", [])
            self._meta = data.get("meta", [{}] * len(self._doc_keys))
            logger.info("BM25 索引加载成功: %d 条", len(self._doc_keys))
            return True

    # ── 属性 ──

    @property
    def doc_count(self) -> int:
        return len(self._doc_keys)

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None
