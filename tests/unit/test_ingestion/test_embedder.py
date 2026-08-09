"""测试 BatchEmbedder — 双层缓存 + 嵌入降级."""

import hashlib
import importlib
import inspect

import numpy as np
import pytest

from src.ingestion.embedder import BatchEmbedder, embedding_text
from src.ingestion.embedding_cache import EmbeddingCache
from src.retriever.vector_store import Chunk


@pytest.fixture
def fake_embed_client():
    """确定性嵌入 stub — 不依赖本地 BGE-M3 / torch 设备状态."""
    class _FakeEmbedClient:
        def embed(self, texts):
            vectors = []
            for text in texts:
                digest = hashlib.sha256(text.encode()).digest()
                vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
                vec = np.tile(vec, (1024 + len(vec) - 1) // len(vec))[:1024]
                vectors.append(vec / (np.linalg.norm(vec) + 1e-8))
            return vectors
    return _FakeEmbedClient()


class TestBatchEmbedderInit:
    """构造与参数."""

    def test_default_init(self):
        embedder = BatchEmbedder()
        assert embedder._batch_size == 32
        assert embedder._sqlite_cache is None
        assert embedder._cache_dir is None

    def test_custom_batch_size(self):
        embedder = BatchEmbedder(batch_size=16)
        assert embedder._batch_size == 16

    def test_with_sqlite_cache(self):
        cache = EmbeddingCache()
        embedder = BatchEmbedder(sqlite_cache=cache)
        assert embedder._sqlite_cache is cache


class TestBatchEmbedderEmpty:
    """空输入处理."""

    def test_embed_batch_empty(self):
        embedder = BatchEmbedder()
        result = embedder.embed_batch([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 1024)

    def test_embed_batch_single(self, fake_embed_client):
        """单条文本嵌入 (需要 BGE-M3)."""
        embedder = BatchEmbedder(batch_size=4)
        embedder._llm_client = fake_embed_client
        result = embedder.embed_batch(["PDU Session Establishment procedure"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1024)
        assert result.dtype == np.float32

    def test_embed_single(self, fake_embed_client):
        embedder = BatchEmbedder(batch_size=4)
        embedder._llm_client = fake_embed_client
        result = embedder.embed_single("Test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_embed_batch_consistency(self, fake_embed_client):
        """同一文本两次嵌入应返回相同维度."""
        embedder = BatchEmbedder(batch_size=4)
        embedder._llm_client = fake_embed_client
        r1 = embedder.embed_single("NR cell search procedure")
        r2 = embedder.embed_single("NR cell search procedure")
        assert r1.shape == r2.shape == (1024,)


class TestCacheIntegration:
    """缓存集成测试 — 验证 SQLite 缓存命中路径."""

    def test_cache_hit_on_second_call(self, tmp_path, fake_embed_client):
        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db")
        embedder = BatchEmbedder(batch_size=4, sqlite_cache=cache)
        embedder._llm_client = fake_embed_client
        texts = ["PDU Session Resource Setup Request Transfer"]

        # 首次嵌入
        e1 = embedder.embed_batch(texts)
        assert e1.shape == (1, 1024)

        # 验证缓存已写入
        cached = cache.get(texts[0])
        assert cached is not None
        np.testing.assert_array_almost_equal(e1[0], cached, decimal=4)

        # 二次调用应从缓存命中
        embedder.embed_batch(texts)
        st = cache.stats()
        assert st["total_entries"] >= 1  # 不应重复写入

    def test_cache_stats_after_embed(self, tmp_path):
        """直接测试缓存统计 — 不依赖嵌入模型."""
        cache = EmbeddingCache(db_path=tmp_path / "test_stats.db")
        texts = [f"stats test chunk {i}" for i in range(5)]
        embs = np.random.randn(5, 1024).astype(np.float32)

        count = cache.put_batch(texts, embs)
        assert count == 5

        st = cache.stats()
        assert st["total_entries"] == 5
        assert st["size_bytes"] > 0
        assert st["size_mb"] > 0
        assert "test_stats.db" in st["db_path"]

    def test_cache_clear(self, tmp_path, fake_embed_client):
        cache = EmbeddingCache(db_path=tmp_path / "test_cache3.db")
        embedder = BatchEmbedder(batch_size=4, sqlite_cache=cache)
        embedder._llm_client = fake_embed_client
        embedder.embed_batch(["test"])
        assert cache.stats()["total_entries"] == 1
        removed = cache.clear()
        assert removed == 1
        assert cache.stats()["total_entries"] == 0


class TestCacheMethods:
    """EmbeddingCache 内部方法."""

    def test_text_key_deterministic(self):
        key1 = EmbeddingCache.text_key("hello")
        key2 = EmbeddingCache.text_key("hello")
        assert key1 == key2
        assert len(key1) == 64  # SHA256

    def test_text_key_case_sensitive(self):
        k1 = EmbeddingCache.text_key("Hello")
        k2 = EmbeddingCache.text_key("hello")
        assert k1 != k2

    def test_get_miss(self):
        cache = EmbeddingCache()
        result = cache.get("nonexistent text 12345")
        assert result is None

    def test_put_get_roundtrip(self, tmp_path):
        cache = EmbeddingCache(db_path=tmp_path / "test_roundtrip.db")
        emb = np.random.randn(1024).astype(np.float32)
        cache.put("roundtrip test text", emb)
        retrieved = cache.get("roundtrip test text")
        assert retrieved is not None
        np.testing.assert_array_equal(emb, retrieved)

    def test_batch_operations(self, tmp_path):
        cache = EmbeddingCache(db_path=tmp_path / "test_batch.db")
        texts = [f"batch text {i}" for i in range(10)]
        embs = np.random.randn(10, 1024).astype(np.float32)

        count = cache.put_batch(texts, embs)
        assert count == 10

        hits = cache.get_batch(texts)
        assert len(hits) == 10
        for i, text in enumerate(texts):
            key = cache.text_key(text)
            assert key in hits
            np.testing.assert_array_equal(embs[i], hits[key])

    def test_batch_mismatch_raises(self, tmp_path):
        cache = EmbeddingCache(db_path=tmp_path / "test_mismatch.db")
        with pytest.raises(ValueError, match="数量不匹配"):
            cache.put_batch(["a", "b"], np.random.randn(3, 1024))

    def test_stats_empty(self):
        cache = EmbeddingCache()
        st = cache.stats()
        assert st["total_entries"] >= 0
        assert "size_mb" in st
        assert "db_path" in st

    def test_clear_returns_count(self, tmp_path):
        cache = EmbeddingCache(db_path=tmp_path / "test_clear.db")
        cache.put("text1", np.random.randn(1024).astype(np.float32))
        cache.put("text2", np.random.randn(1024).astype(np.float32))
        removed = cache.clear()
        assert removed == 2
        assert cache.stats()["total_entries"] == 0


# ── embedding_text() 唯一真源: 嵌入文本构成为纯正文 ──

EMBEDDING_TEXT_PATH_MODULES = [
    "scripts.bulk_ingest",
    "src.ingestion.orchestrator",
    "src.ingestion.incremental",
    "scripts.reindex_bge_m3",
]


class TestEmbeddingTextHelper:
    """embedding_text() 必须返回纯正文 (chunk.text), 不拼接标题/章节路径."""

    def test_returns_pure_body_with_section_fields(self):
        chunk = Chunk(
            text="The UE shall transmit at the configured power level.",
            section_title="UE behaviour",
            section_path="7 Uplink Power control > 7.1 PUSCH > 7.1.1 UE behaviour",
        )
        assert embedding_text(chunk) == chunk.text

    def test_returns_pure_body_without_section_fields(self):
        body = "Independent chunk text"
        assert embedding_text(Chunk(text=body)) == body

    def test_returns_empty_for_empty_text(self):
        assert embedding_text(Chunk(text="", section_title="X", section_path="A > B")) == ""

    def test_preserves_whitespace(self):
        body = "  padded body  "
        assert embedding_text(Chunk(text=body)) == body


class TestAllIngestionPathsUseSingleSourceHelper:
    """四条摄入路径全部路由到 embedding_text() —— 彼此一致、均为纯正文."""

    @pytest.mark.parametrize("module_path", EMBEDDING_TEXT_PATH_MODULES)
    def test_path_binds_single_source_helper(self, module_path):
        mod = importlib.import_module(module_path)
        # 身份一致 ⇒ 四条路径产出的嵌入文本构成彼此一致 (唯一真源)
        assert mod.embedding_text is embedding_text

    @pytest.mark.parametrize("module_path", EMBEDDING_TEXT_PATH_MODULES)
    def test_path_embedding_text_is_pure_body(self, module_path):
        mod = importlib.import_module(module_path)
        body = "RRC Reconfiguration complete message body."
        chunk = Chunk(
            text=body,
            section_title="RRC Reconfiguration",
            section_path="5.3 RRC Reconfiguration > 5.3.1 General",
        )
        # 产出 === 纯 c.text, 无标题/路径前缀
        assert mod.embedding_text(chunk) == body

    @pytest.mark.parametrize("module_path", EMBEDDING_TEXT_PATH_MODULES)
    def test_path_has_no_leftover_title_concat(self, module_path):
        # 回归防护: 旧拼接 f"{c.section_title} {c.section_path} {c.text}" 必须已移除
        mod = importlib.import_module(module_path)
        assert "c.section_title} {c.section_path}" not in inspect.getsource(mod)
