"""预计算嵌入数据导入 — .npy + Documents.db → VectorStore.

.. deprecated:: 2026-07-14
    此模块已废弃。HuggingFace 3GPP-R18 预计算 .npy 数据集不再使用，
    当前系统统一使用 `bulk_ingest.py` 从本地 DOCX 直接嵌入。
    保留此文件仅作历史参考，后续版本将删除。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import warnings
from typing import Any

import numpy as np

from src.config import settings
from src.retriever.vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)

warnings.warn(
    "precomputed_loader 已废弃 (2026-07-14)。"
    " 当前系统使用 bulk_ingest.py 从本地 DOCX 直接嵌入，不再依赖 HuggingFace 预计算数据集。"
    " 此模块保留仅作历史参考，后续版本将删除。",
    DeprecationWarning,
    stacklevel=2,
)


class PrecomputedLoader:
    """加载 3GPP-R18 预计算数据集，导入向量数据库.

    数据来源: HuggingFace netop/Embeddings3GPP-R18
    数据格式: 每 Series 一个 .npy (1024-dim float64)，Documents.db 存原始文本.
    """

    # Telco-RAG 的原始分块参数（必须一致才能对齐 .npy 行）
    CHUNK_SIZE = 500       # 字符数
    CHUNK_OVERLAP = 25     # 重叠字符数
    WORD_SPLIT = True      # 词边界对齐

    # 3GPP 章节标题正则 (e.g. "5.1.2  PDU Session Establishment")
    SECTION_RE = re.compile(
        r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE
    )

    def __init__(self, vector_store: VectorStore):
        self._store = vector_store
        self._data_path = settings.pre_computed_abs_path
        self._embeddings_dir = self._data_path / "Embeddings"
        self._db_path = self._data_path / "Documents.db"

    # ── 主流程 ──

    def load_all(self, series_range: list[int] | None = None) -> int:
        """加载全部预计算数据，返回导入的 chunk 总数.

        Args:
            series_range: 要加载的 Series 列表，None=全部.

        Returns:
            成功导入的 chunk 总数.
        """
        if series_range is None:
            series_range = settings.pre_computed_series_list or list(range(21, 39))

        # 1. 加载所有文档
        logger.info("加载文档数据库: %s", self._db_path)
        documents: dict[int, list[dict[str, Any]]] = self._load_documents(series_range)

        # 2. 逐 Series 处理
        total = 0
        for series_num in series_range:
            docs = documents.get(series_num, [])
            if not docs:
                logger.debug("Series %d: 无文档，跳过", series_num)
                continue

            count = self._load_series(series_num, docs)
            total += count
            logger.info("Series %d: 导入 %d 条", series_num, count)

        return total

    def _load_series(
        self, series_num: int, documents: list[dict[str, Any]]
    ) -> int:
        """加载单个 Series 的嵌入和文档.

        Args:
            series_num: 3GPP Series 编号 (21-38).
            documents: 该 Series 下的文档列表 (按文件名排序).

        Returns:
            导入的 chunk 数量.
        """
        # 加载 .npy 嵌入
        npy_path = self._embeddings_dir / f"EmbeddingsSeries{series_num}.npy"
        if not npy_path.exists():
            logger.warning("%s 不存在，跳过", npy_path)
            return 0

        embeddings = np.load(npy_path)  # shape: (N, 1024), dtype: float64
        if embeddings.shape[0] == 0:
            logger.info("Series %d: 空嵌入，跳过", series_num)
            return 0

        # 对文档分块
        text_chunks: list[Chunk] = []
        total_chars = 0
        for doc in documents:
            chunks = self._chunk_document(doc)
            text_chunks.extend(chunks)
            total_chars += len(doc.get("text", ""))

        # 对齐检查
        npy_rows = embeddings.shape[0]
        chunk_count = len(text_chunks)

        if npy_rows != chunk_count:
            logger.warning(
                "Series %d: 嵌入行数(%d) != 分块数(%d)，差异=%d，将用文档再嵌入替代",
                series_num, npy_rows, chunk_count, npy_rows - chunk_count,
            )
            # 不对齐时仍尝试按 min 数量导入
            effective = min(npy_rows, chunk_count)
        else:
            effective = npy_rows
            logger.debug(
                "Series %d: 完美对齐 %d chunks (avg %.0f chars/chunk)",
                series_num, effective, total_chars / effective if effective else 0,
            )

        # 组装 Chunk
        chunks_to_insert: list[Chunk] = []
        for i in range(effective):
            c = text_chunks[i]
            c.embedding = embeddings[i]  # 注入预计算嵌入
            chunks_to_insert.append(c)

        # 批量写入
        batch_size = 500
        inserted = 0
        for start in range(0, len(chunks_to_insert), batch_size):
            batch = chunks_to_insert[start : start + batch_size]
            inserted += self._store.insert(batch)

        return inserted

    # ── 文档加载 ──

    def _load_documents(
        self, series_range: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        """从 Documents.db 加载指定 Series 的文档.

        Returns:
            {series_num: [doc_dict, ...]}，按 doc_id 排序.
        """
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        docs_by_series: dict[int, list[dict[str, Any]]] = {
            s: [] for s in series_range
        }

        cursor.execute("SELECT id, data FROM Standard")
        for row in cursor.fetchall():
            doc_id: str = row[0]
            data_json: str = row[1]
            data = json.loads(data_json)

            # 从文件名提取 Series (前两位数字)
            series = self._extract_series(doc_id)
            if series not in series_range:
                continue

            data["_doc_id"] = doc_id
            data["_series"] = series
            docs_by_series[series].append(data)

        conn.close()

        # 各 Series 内按文件名排序 (保证与嵌入顺序一致)
        for s in docs_by_series:
            docs_by_series[s].sort(key=lambda d: d["_doc_id"])

        total = sum(len(v) for v in docs_by_series.values())
        logger.info("从数据库加载 %d 篇文档，覆盖 %d 个 Series", total, len(docs_by_series))
        return docs_by_series

    @staticmethod
    def _extract_series(doc_id: str) -> int:
        """从文件名提取 Series 编号."""
        prefix = doc_id[:2]
        if prefix.isdigit():
            return int(prefix)
        # re 开头的文件 (e.g. "release-...") → series 0
        return 0

    # ── 文档分块 ──

    def _chunk_document(self, doc: dict[str, Any]) -> list[Chunk]:
        """使用 Telco-RAG 的 custom_text_splitter 策略对单篇文档分块.

        同时解析章节标题，为每个 chunk 附加 parent_section_id / parent_title.
        """
        text = doc.get("text", "")
        doc_id = doc.get("_doc_id", "")
        series = doc.get("_series", 0)
        spec_number = self._extract_spec_number(doc_id, text)
        release = self._extract_release(text)

        raw_chunks = self._custom_text_splitter(
            text, self.CHUNK_SIZE, self.CHUNK_OVERLAP, self.WORD_SPLIT
        )

        # 解析章节标题 → 构建位置映射 (字符位置 → 章节信息)
        section_map = self._build_section_map(text)

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(raw_chunks):
            # 在原文中定位 chunk 起始位置
            pos = text.find(chunk_text) if idx == 0 else text.find(chunk_text)
            parent_sec_id, parent_title = "", ""
            if pos >= 0 and section_map:
                # 找到最近的上方章节标题
                parent_sec_id, parent_title = self._find_parent_section(section_map, pos)

            chunks.append(Chunk(
                text=chunk_text,
                embedding=None,  # 由调用方注入
                doc_id=doc_id,
                series=series,
                spec_number=spec_number,
                release=release,
                parent_section_id=parent_sec_id,
                parent_title=parent_title,
                chunk_index=idx,
            ))

        return chunks

    @staticmethod
    def _custom_text_splitter(
        text: str, chunk_size: int, chunk_overlap: int, word_split: bool = True
    ) -> list[str]:
        """Telco-RAG 原始分块算法 (逐字符分块 + 词边界对齐)."""
        chunks: list[str] = []
        start = 0
        separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')

        while start < len(text) - chunk_overlap:
            end = min(start + chunk_size, len(text))

            if word_split:
                match = separators_pattern.search(text, end)
                if match:
                    end = match.end()

            if end == start:
                end = start + 1

            chunks.append(text[start:end])
            start = end - chunk_overlap

            if word_split:
                # 从下一个词边界开始 (避免截断)
                match = separators_pattern.search(text, start - 1)
                if match:
                    start = match.start() + 1

            if start < 0:
                start = 0

        return chunks

    @staticmethod
    def _build_section_map(text: str) -> list[tuple[int, str, str]]:
        """解析文档中的章节标题，返回 [(起始位置, 编号, 标题), ...]."""
        sections: list[tuple[int, str, str]] = []
        for m in re.finditer(
            r"^(\d+(?:\.\d+)*)\s+(.{5,120})$", text, re.MULTILINE
        ):
            sec_id = m.group(1)
            sec_title = m.group(2).strip()
            # 过滤掉大纲/目录行 (全是数字编号的)
            if len(sec_id.split(".")) <= 4:
                sections.append((m.start(), sec_id, sec_title))
        return sections

    @staticmethod
    def _find_parent_section(
        section_map: list[tuple[int, str, str]], pos: int
    ) -> tuple[str, str]:
        """找到给定字符位置最近的父章节."""
        parent_id, parent_title = "", ""
        for sec_pos, sec_id, sec_title in section_map:
            if sec_pos <= pos:
                parent_id, parent_title = sec_id, sec_title
            else:
                break
        return parent_id, parent_title

    @staticmethod
    def _extract_spec_number(doc_id: str, text: str) -> str:
        """从文档 ID 或文本中提取规范编号 (如 '22.101')."""
        # 从 doc_id 提取: 截取前5位数字字符
        digits = ""
        for ch in doc_id:
            if ch.isdigit():
                digits += ch
            elif digits:
                break

        if len(digits) >= 5:
            return f"{digits[0:2]}.{digits[2:5]}"

        # 从文本第一行解析
        first_line = text.split("\n", 1)[0] if text else ""
        m = re.search(r"TS\s*(\d{2}\.\d{3})", first_line, re.IGNORECASE)
        if m:
            return m.group(1)

        return ""

    @staticmethod
    def _extract_release(text: str) -> str:
        """从文档文本中提取 Release 版本."""
        m = re.search(r"Release\s+(\d{2})\)", text[:500], re.IGNORECASE)
        if m:
            return f"R{m.group(1)}"
        return ""

    # ── 汇总嵌入 ──

    def load_summaries(self) -> int:
        """加载汇总嵌入 (EmbeddingsSummaries.npy)."""
        npy_path = self._embeddings_dir / "EmbeddingsSummaries.npy"
        if not npy_path.exists():
            logger.warning("Summaries 文件不存在")
            return 0

        embeddings = np.load(npy_path)
        logger.info("加载 Summaries: %d 条", embeddings.shape[0])

        # Summaries 没有对应的文档文本，用占位符
        chunks = []
        for i in range(embeddings.shape[0]):
            chunks.append(Chunk(
                text=f"[Summary #{i}]",
                embedding=embeddings[i],
                doc_id="summaries",
                series=0,
                spec_number="",
                release="R18",
            ))

        inserted = self._store.insert(chunks)
        logger.info("Summaries 导入完成: %d 条", inserted)
        return inserted
