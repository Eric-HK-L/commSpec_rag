"""增量索引 — 检测文件变更，仅处理新增/修改的文档.

支持:
  - 文件级 hash 检测 (MD5)
  - 时间戳检测
  - 状态持久化: data/.index_state.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.retriever.vector_store import VectorStore
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """单个文档的索引记录."""
    spec_number: str
    release: str
    source_hash: str
    chunk_count: int
    indexed_at: float  # Unix timestamp


class IncrementalIndexer:
    """增量索引器 — 仅处理变更文档."""

    def __init__(
        self,
        vector_store: VectorStore,
        state_path: str | None = None,
        data_dir: str | None = None,
    ):
        self._store = vector_store
        self._state_path = Path(state_path) if state_path else settings.data_abs_dir / ".index_state.json"
        self._data_dir = Path(data_dir) if data_dir else settings.data_abs_dir / "processed"
        self._state: dict[str, IndexEntry] = {}

    # ── 公共 API ──

    def load_state(self) -> dict[str, IndexEntry]:
        """加载索引状态."""
        if self._state_path.exists():
            with open(self._state_path) as f:
                raw = json.load(f)
                self._state = {
                    k: IndexEntry(**v) for k, v in raw.items()
                }
        else:
            self._state = {}
        logger.info("加载索引状态: %d 个文档", len(self._state))
        return self._state

    def save_state(self) -> None:
        """持久化索引状态."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._state.items()}
        with open(self._state_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def detect_changes(self) -> tuple[list[str], list[str], list[str]]:
        """检测文件变更.

        Returns:
            (新增文件, 修改文件, 删除文件) 的绝对路径列表.
        """
        current_files = self._scan_docx_files()

        new_files: list[str] = []
        modified_files: list[str] = []
        deleted_files: list[str] = []

        current_keys = {self._file_key(p): p for p in current_files}

        # 新增 + 修改
        for key, path in current_keys.items():
            current_hash = self._hash_file(path)
            if key not in self._state:
                new_files.append(str(path))
            elif self._state[key].source_hash != current_hash:
                modified_files.append(str(path))

        # 删除
        for key in self._state:
            if key not in current_keys:
                deleted_files.append(key)

        logger.info(
            "变更检测: +%d ~%d -%d",
            len(new_files), len(modified_files), len(deleted_files),
        )
        return new_files, modified_files, deleted_files

    def process_incremental(
        self,
        new_files: list[str],
        modified_files: list[str],
        deleted_files: list[str],
    ) -> dict[str, int]:
        """处理增量变更.

        Returns:
            {"inserted": N, "deleted": N}.
        """
        from src.ingestion.embedder import BatchEmbedder
        from src.ingestion.extractor import PandocExtractor
        from src.ingestion.splitter import HeaderAwareSplitter

        extractor = PandocExtractor()
        splitter = HeaderAwareSplitter()
        embedder = BatchEmbedder(cache_dir=str(settings.data_abs_dir / "cache" / "embeddings"))

        stats = {"inserted": 0, "deleted": 0}

        # 处理新增文件
        for fp in new_files:
            result = extractor.extract_file(fp)
            if not result.markdown:
                continue

            doc_meta = {
                "doc_id": Path(fp).stem,
                "spec_number": result.spec_number,
                "release": result.release,
            }
            chunks = splitter.split_document(result.markdown, doc_meta)

            texts = [c.text for c in chunks]
            embeddings = embedder.embed_batch(texts)
            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            inserted = self._store.insert(chunks)
            stats["inserted"] += inserted

            # 更新状态
            key = self._file_key(fp)
            self._state[key] = IndexEntry(
                spec_number=result.spec_number,
                release=result.release,
                source_hash=self._hash_file(Path(fp)),
                chunk_count=len(chunks),
                indexed_at=time.time(),
            )

        # 处理修改文件 → 删除旧 + 插入新
        for fp in modified_files:
            key = self._file_key(fp)
            old_entry = self._state.get(key)
            if old_entry:
                self._store.delete_by_filter(f'spec_number == "{old_entry.spec_number}"')
                stats["deleted"] += old_entry.chunk_count

            # 重新处理 (同新增逻辑)
            result = extractor.extract_file(fp)
            if not result.markdown:
                continue

            chunks = splitter.split_document(result.markdown, {"doc_id": Path(fp).stem, "spec_number": result.spec_number, "release": result.release})
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_batch(texts)
            for c, emb in zip(chunks, embeddings):
                c.embedding = emb

            inserted = self._store.insert(chunks)
            stats["inserted"] += inserted

            self._state[key] = IndexEntry(
                spec_number=result.spec_number,
                release=result.release,
                source_hash=self._hash_file(Path(fp)),
                chunk_count=len(chunks),
                indexed_at=time.time(),
            )

        # 处理删除文件
        for key in deleted_files:
            entry = self._state.pop(key, None)
            if entry:
                self._store.delete_by_filter(f'spec_number == "{entry.spec_number}"')
                stats["deleted"] += entry.chunk_count

        self.save_state()
        return stats

    # ── 工具方法 ──

    def _scan_docx_files(self) -> list[Path]:
        """扫描所有 DOCX 文件."""
        if not self._data_dir.exists():
            return []
        return sorted(self._data_dir.rglob("*.docx"))

    @staticmethod
    def _file_key(path: str | Path) -> str:
        """生成文件唯一键 (相对路径)."""
        return str(Path(path).name)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """计算文件 MD5."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
