"""embedding_cache.py 单元测试 — SHA256 哈希 / 纯函数."""

import hashlib

from src.ingestion.embedding_cache import EmbeddingCache


class TestTextKey:

    def test_deterministic(self):
        k1 = EmbeddingCache.text_key("hello")
        k2 = EmbeddingCache.text_key("hello")
        assert k1 == k2

    def test_different_texts(self):
        k1 = EmbeddingCache.text_key("hello")
        k2 = EmbeddingCache.text_key("world")
        assert k1 != k2

    def test_sha256_format(self):
        key = EmbeddingCache.text_key("test")
        assert len(key) == 64  # SHA256 hex 长度
        assert all(c in "0123456789abcdef" for c in key)

    def test_matches_raw_sha256(self):
        expected = hashlib.sha256("test".encode("utf-8")).hexdigest()
        assert EmbeddingCache.text_key("test") == expected

    def test_unicode(self):
        key = EmbeddingCache.text_key("PDU会话建立")
        assert len(key) == 64

    def test_empty_string(self):
        key = EmbeddingCache.text_key("")
        assert len(key) == 64
        assert key == hashlib.sha256(b"").hexdigest()
