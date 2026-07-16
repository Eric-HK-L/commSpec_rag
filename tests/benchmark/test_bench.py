"""RAG 系统基准测试 — 吞吐/延迟/缓存命中率.

运行:
  python -m pytest tests/benchmark/ -v
  python -m pytest tests/benchmark/test_bench.py -k "test_cache" -v
"""

import time

import numpy as np
import pytest


# ── 嵌入缓存基准 ──

class TestEmbeddingCacheBenchmark:
    """EmbeddingCache 吞吐与延迟."""

    def test_batch_write_throughput(self, tmp_path):
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(db_path=tmp_path / "bench_cache.db")
        n = 500
        texts = [f"benchmark text chunk number {i} with some 3GPP terminology PDU NAS RRC" for i in range(n)]
        embs = np.random.randn(n, 1024).astype(np.float32)

        t0 = time.perf_counter()
        count = cache.put_batch(texts, embs)
        elapsed = time.perf_counter() - t0

        assert count == n
        tps = n / elapsed if elapsed > 0 else float("inf")
        print(f"\n  写入 {n} 条: {elapsed*1000:.1f}ms ({tps:.0f} t/s)")
        assert tps > 100  # 至少 100 t/s

    def test_batch_read_throughput(self, tmp_path):
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(db_path=tmp_path / "bench_read.db")
        n = 500
        texts = [f"read test text {i} for benchmark" for i in range(n)]
        embs = np.random.randn(n, 1024).astype(np.float32)
        cache.put_batch(texts, embs)

        t0 = time.perf_counter()
        hits = cache.get_batch(texts)
        elapsed = time.perf_counter() - t0

        assert len(hits) == n
        tps = n / elapsed if elapsed > 0 else float("inf")
        print(f"\n  读取 {n} 条: {elapsed*1000:.1f}ms ({tps:.0f} t/s)")
        assert tps > 500  # 读取应比写入快

    def test_cache_hit_ratio_with_repeats(self, tmp_path):
        """重复摄入场景的缓存命中率."""
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(db_path=tmp_path / "bench_hit.db")
        # 100 条唯一文本 + 400 条重复 (80% 重复率)
        unique = [f"unique text {i}" for i in range(100)]
        repeated = unique * 4  # 400 条
        all_texts = unique + repeated
        embs = np.random.randn(len(unique), 1024).astype(np.float32)

        # 首次写入 unique
        cache.put_batch(unique, embs)

        # 查询全部 — get_batch 返回按 hash 去重的 dict
        hits = cache.get_batch(all_texts)
        # 500 条文本中只有 100 个唯一 hash，缓存应命中全部唯一文本
        unique_hit_rate = len(hits) / len(unique) * 100
        print(f"\n  唯一文本命中率: {unique_hit_rate:.1f}% ({len(hits)}/{len(unique)})")
        assert unique_hit_rate == 100  # 所有唯一文本都应命中

    def test_cache_size_estimate(self, tmp_path):
        """估算缓存存储开销."""
        from src.ingestion.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(db_path=tmp_path / "bench_size.db")
        n = 200
        texts = [f"size benchmark chunk {i:04d}" for i in range(n)]
        embs = np.random.randn(n, 1024).astype(np.float32)
        cache.put_batch(texts, embs)

        st = cache.stats()
        bytes_per_entry = st["size_bytes"] / st["total_entries"]
        print(f"\n  存储 {n} 条: {st['size_mb']:.2f} MB ({bytes_per_entry:.0f} B/条)")
        assert bytes_per_entry < 5000  # 每条约 4KB


# ── 哈希性能基准 ──

class TestHashBenchmark:
    """SHA256 计算吞吐."""

    def test_sha256_throughput(self):
        import hashlib
        n = 10000
        texts = [f"hash benchmark text chunk {i:05d} with protocol terminology PDU Session NAS RRC NGAP" for i in range(n)]

        t0 = time.perf_counter()
        for t in texts:
            hashlib.sha256(t.encode("utf-8")).hexdigest()
        elapsed = time.perf_counter() - t0

        tps = n / elapsed if elapsed > 0 else float("inf")
        print(f"\n  SHA256 {n} 条: {elapsed*1000:.1f}ms ({tps:.0f} hash/s)")
        assert tps > 10000  # 至少 10K hash/s


# ── 查询缓存基准 ──

class TestQueryCacheBenchmark:
    """查询级 LRU 缓存性能."""

    def test_ttl_cache_lookup(self):
        from cachetools import TTLCache
        import hashlib
        cache = TTLCache(maxsize=1000, ttl=3600)

        # 预填充
        for i in range(500):
            cache[f"key_{i}"] = f"value_{i}"

        # 基准: 10000 次查询
        n = 10000
        t0 = time.perf_counter()
        hits = 0
        for i in range(n):
            key = hashlib.md5(f"test query number {i % 1000}".encode()).hexdigest()
            if key in cache:
                hits += 1
        elapsed = time.perf_counter() - t0

        ops = n / elapsed if elapsed > 0 else float("inf")
        print(f"\n  TTLCache {n} 次查找: {elapsed*1000:.1f}ms ({ops:.0f} ops/s), 命中 {hits}")
        assert ops > 50000


# ── 检索基准 (无 Milvus 的轻量级模拟) ──

class TestSearchBenchmark:
    """模拟检索操作的延迟基准."""

    def test_numpy_similarity_throughput(self):
        """模拟 Dense 相似度计算 (余弦)."""
        n_queries = 100
        n_docs = 5000
        dim = 1024

        queries = np.random.randn(n_queries, dim).astype(np.float32)
        docs = np.random.randn(n_docs, dim).astype(np.float32)

        # 归一化
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

        t0 = time.perf_counter()
        for q in queries:
            scores = np.dot(docs, q)  # cosine = dot (normed)
            top_k = np.argpartition(-scores, 20)[:20]
            top_k = top_k[np.argsort(-scores[top_k])]
        elapsed = time.perf_counter() - t0

        avg_ms = elapsed / n_queries * 1000
        print(f"\n  模拟检索 {n_queries} 查询 × {n_docs} 文档: {elapsed*1000:.1f}ms ({avg_ms:.1f}ms/query)")
        assert avg_ms < 50  # 纯 numpy 应很快

    def test_rrf_fusion_benchmark(self):
        """RRF 融合计算开销."""
        n = 1000
        dense_ranks = np.random.permutation(n)[:100]
        bm25_ranks = np.random.permutation(n)[:100]

        t0 = time.perf_counter()
        for _ in range(1000):
            _k = 60
            dense_scores = {r: 1.0 / (_k + i + 1) for i, r in enumerate(dense_ranks)}
            bm25_scores = {r: 1.0 / (_k + i + 1) for i, r in enumerate(bm25_ranks)}
            all_ids = set(dense_scores) | set(bm25_scores)
            _fused = {rid: dense_scores.get(rid, 0) + bm25_scores.get(rid, 0) for rid in all_ids}
        elapsed = time.perf_counter() - t0

        ops = 1000 / elapsed if elapsed > 0 else float("inf")
        print(f"\n  RRF 融合 1000 次: {elapsed*1000:.1f}ms ({ops:.0f} ops/s)")
        assert ops > 200
