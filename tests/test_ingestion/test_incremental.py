"""incremental.py 单元测试 — 纯函数: _file_key, _hash_file, IndexEntry."""

import hashlib
import tempfile
from pathlib import Path

from src.ingestion.incremental import IncrementalIndexer, IndexEntry


class TestFileKey:
    """_file_key — 文件唯一键生成 (静态方法)."""

    def test_stem_only(self):
        key = IncrementalIndexer._file_key("data/documents/38.413.docx")
        assert key == "38.413.docx"

    def test_path_object(self):
        key = IncrementalIndexer._file_key(Path("/abs/path/doc.docx"))
        assert key == "doc.docx"

    def test_str_no_path(self):
        key = IncrementalIndexer._file_key("simple.docx")
        assert key == "simple.docx"


class TestHashFile:
    """_hash_file — MD5 文件哈希."""

    def test_consistent_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content for hashing")
            tmp = f.name

        try:
            h1 = IncrementalIndexer._hash_file(Path(tmp))
            h2 = IncrementalIndexer._hash_file(Path(tmp))
            assert h1 == h2
            assert len(h1) == 32  # MD5 hex
        finally:
            Path(tmp).unlink()

    def test_different_content_different_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content A")
            tmp_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content B")
            tmp_b = f.name

        try:
            h_a = IncrementalIndexer._hash_file(Path(tmp_a))
            h_b = IncrementalIndexer._hash_file(Path(tmp_b))
            assert h_a != h_b
        finally:
            Path(tmp_a).unlink()
            Path(tmp_b).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            tmp = f.name
        try:
            h = IncrementalIndexer._hash_file(Path(tmp))
            expected = hashlib.md5(b"").hexdigest()
            assert h == expected
        finally:
            Path(tmp).unlink()


class TestIndexEntry:
    """IndexEntry — 索引记录 dataclass."""

    def test_create(self):
        entry = IndexEntry(
            spec_number="38.413",
            release="R18",
            source_hash="abc123",
            chunk_count=42,
            indexed_at=1720000000.0,
        )
        assert entry.spec_number == "38.413"
        assert entry.chunk_count == 42

    def test_defaults(self):
        # 所有字段都是必填的 (无默认值)
        entry = IndexEntry(
            spec_number="", release="", source_hash="", chunk_count=0, indexed_at=0.0,
        )
        assert entry.chunk_count == 0
