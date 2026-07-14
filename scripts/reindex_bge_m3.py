#!/usr/bin/env python3
"""BGE-M3 全量重索引 + bge-large-en-v1.5 横向对比。

流程:
  1. 从现有 Milvus collection (TeleComm_specs) 读取全部 chunk 文本+元数据
  2. MPSChunkedEmbedder("BAAI/bge-m3") 重嵌全部 chunk → 测速/内存
  3. 创建新 collection TeleComm_specs_m3 → 入库
  4. 构建 BM25 索引
  5. 跑评测框架对比 Recall@k / MRR / NDCG

用法:
  python scripts/reindex_bge_m3.py                        # 完整流程
  python scripts/reindex_bge_m3.py --embed-only           # 仅嵌入 (不入库)
  python scripts/reindex_bge_m3.py --eval-only            # 仅评测 (需两 collection 均已就绪)
  python scripts/reindex_bge_m3.py --dry-run              # 预览 (不执行)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.mps_embedder import MPSChunkedEmbedder  # noqa: E402
from src.retriever.milvus_store import MilvusStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_m3")

# ── 常量 ──
BGE_V1_COLLECTION = "TeleComm_specs"
BGE_M3_COLLECTION = "TeleComm_specs_m3"
BGE_M3_MODEL = "BAAI/bge-m3"


# ── 工具 ──

def read_chunks_from_collection(store: MilvusStore, label: str) -> list[dict]:
    """从 Milvus collection 读取全部 chunk 文本和元数据."""
    store.connect()
    store._collection_name = (
        BGE_M3_COLLECTION if label == "m3" else BGE_V1_COLLECTION
    )
    store._collection = None
    store._ensure_connected()

    col = store._collection
    if col is None:
        logger.error("%s collection 不存在", store._collection_name)
        return []

    logger.info("查询 %s 全部 chunks...", store._collection_name)
    try:
        results = col.query(
            expr="id >= 0",
            output_fields=[
                "text", "doc_id", "series", "spec_number",
                "release", "parent_section_id", "parent_title", "chunk_index",
            ],
            limit=300000,
        )
    except Exception as e:
        logger.error("查询失败: %s", e)
        return []

    logger.info("  %d chunks 已读取", len(results))
    return results


def build_chunk_objects(rows: list[dict]) -> list:
    """将 Milvus 行转换为 Chunk 对象列表."""
    from src.retriever.vector_store import Chunk
    from src.retriever.milvus_store import VARCHAR_MAX, _safe_truncate_bytes

    chunks = []
    for r in rows:
        text = str(r.get("text", ""))
        chunks.append(Chunk(
            text=_safe_truncate_bytes(text, VARCHAR_MAX),
            doc_id=str(r.get("doc_id", "")),
            series=int(r.get("series", 0)),
            spec_number=str(r.get("spec_number", "")),
            release=str(r.get("release", "")),
            parent_section_id=str(r.get("parent_section_id", "")),
            parent_title=str(r.get("parent_title", "")),
            chunk_index=int(r.get("chunk_index", 0)),
        ))
    return chunks


# ── 对比主流程 ──

def benchmark_embed(
    texts: list[str],
    model_name: str,
    label: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, dict]:
    """嵌入基准测试：测速、测内存、验证 MPS spawn 正确性。

    Returns:
        (embeddings, metrics_dict)
    """
    metrics: dict[str, Any] = {"model": model_name, "label": label}
    n = len(texts)
    if n == 0:
        return np.empty((0, 1024), dtype=np.float32), metrics

    # 模型加载内存
    tracemalloc.start()
    t0 = time.time()

    embedder = MPSChunkedEmbedder(model_name=model_name)
    embeddings = embedder.embed_raw(texts, batch_size=batch_size)

    elapsed = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 验证
    metrics["total_texts"] = n
    metrics["total_batches"] = (n + batch_size - 1) // batch_size
    metrics["total_time_s"] = elapsed
    metrics["texts_per_second"] = n / elapsed if elapsed > 0 else 0
    metrics["peak_memory_mb"] = peak_bytes / 1e6
    metrics["embedding_shape"] = list(embeddings.shape)
    metrics["embedding_dim"] = embeddings.shape[1]
    metrics["nan_count"] = int(np.isnan(embeddings).sum())
    metrics["zero_vector_count"] = int((np.linalg.norm(embeddings, axis=1) < 1e-8).sum())

    logger.info(
        "  %s: %.1fs, %.0f t/s, peak %.0f MB, dim=%d, NaN=%d, zero=%d",
        label, elapsed, metrics["texts_per_second"],
        metrics["peak_memory_mb"], metrics["embedding_dim"],
        metrics["nan_count"], metrics["zero_vector_count"],
    )
    return embeddings, metrics


def compare_models_on_subset(texts: list[str], subset_size: int = 64):
    """在小子集上对比两模型速度 (快速验证)."""
    subset = texts[:subset_size] if len(texts) > subset_size else texts
    logger.info("── 快速基准 (子集 %d texts) ──", len(subset))

    results = {}
    for model_name, label in [
        (settings.local_embedding_model, "bge-large-v1.5"),
        (BGE_M3_MODEL, "BGE-M3"),
    ]:
        _, metrics = benchmark_embed(subset, model_name, label)
        results[label] = metrics

    # 打印对比表
    print(f"\n  {'指标':<30} {'bge-large-v1.5':<20} {'BGE-M3':<20}")
    print(f"  {'─' * 30} {'─' * 20} {'─' * 20}")
    for key, fmt in [
        ("total_time_s", "{:<20.1f}"),
        ("texts_per_second", "{:<20.0f}"),
        ("peak_memory_mb", "{:<20.0f}"),
        ("embedding_dim", "{:<20}"),
    ]:
        v1 = results.get("bge-large-v1.5", {}).get(key, "N/A")
        v3 = results.get("BGE-M3", {}).get(key, "N/A")
        label_map = {
            "total_time_s": "耗时 (s)",
            "texts_per_second": "吞吐 (t/s)",
            "peak_memory_mb": "峰值内存 (MB)",
            "embedding_dim": "嵌入维度",
        }
        print(f"  {label_map.get(key, key):<30} {fmt.format(v1) if isinstance(fmt, str) else v1} {fmt.format(v3) if isinstance(fmt, str) else v3}")
    print()

    return results


def insert_to_collection(
    chunks: list,
    embeddings: np.ndarray,
    store: MilvusStore,
    label: str,
) -> int:
    """将 chunks + embeddings 入库到指定 collection."""
    from src.retriever.milvus_store import VARCHAR_MAX

    # 切换到目标 collection
    store._collection_name = BGE_M3_COLLECTION
    store._collection = None
    if store._connected:
        store.disconnect()
    store.connect()
    store.create_collection(drop_existing=True)

    # 注入 embedding
    for c, emb in zip(chunks, embeddings):
        c.embedding = emb.astype(np.float32)

    # 批量插入
    total = len(chunks)
    batch_size = 500
    inserted = 0
    t0 = time.time()
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        n = store.insert(batch)
        inserted += n
        if (start // batch_size + 1) % 10 == 0:
            logger.info("  入库进度: %d/%d", inserted, total)

    logger.info(
        "入库完成: %d/%d, %.1fs (%.0f c/s)",
        inserted, total, time.time() - t0,
        total / (time.time() - t0) if time.time() > t0 else 0,
    )

    # BM25
    logger.info("构建 BM25 索引...")
    store.build_bm25(
        [c.text for c in chunks],
        [c.doc_id for c in chunks],
        [c.spec_number for c in chunks],
        [c.chunk_index for c in chunks],
    )
    logger.info("BM25 完成 (%d 条)", len(chunks))

    return inserted


def run_eval_comparison() -> dict:
    """跑评测对比, 输出指标."""
    logger.info("── 评测对比 ──")
    try:
        from tests.eval.metrics import EvalReport, EvalResult, EvalSample, evaluate_one
        from tests.eval.run_eval import load_test_set
    except ImportError as e:
        logger.error("评测模块不可用: %s", e)
        return {}

    test_set_path = str(PROJECT_ROOT / "tests" / "eval" / "test_set.json")
    samples = load_test_set(test_set_path)
    logger.info("测试集: %d 条", len(samples))

    results: dict[str, dict] = {}
    for coll_name, label in [
        (BGE_V1_COLLECTION, "bge-large-v1.5"),
        (BGE_M3_COLLECTION, "BGE-M3"),
    ]:
        logger.info("评测 %s (%s)...", label, coll_name)
        # 为每个 collection 创建独立的 store + retriever
        from src.retriever.search import HybridRetriever
        from src.generator.llm_client import LLMClient

        store = MilvusStore(collection_name=coll_name)
        store.connect()
        store._collection_name = coll_name
        if not store._bm25.is_loaded:
            # 尝试加载 BM25
            bm25_path = settings.vectors_dir / f"bm25_index_{label}.pkl"
            store._bm25._index_path = Path(bm25_path)
            if not store.load_bm25():
                logger.warning("  BM25 未加载, 降级纯 Dense")

        llm = LLMClient()
        retriever = HybridRetriever(store)

        eval_results: list[EvalResult] = []
        for i, sample in enumerate(samples):
            try:
                q_emb = np.array(llm.embed([sample.question])[0], dtype=np.float32)
                retrieval = retriever.search(sample.question, q_emb)
                retrieved_specs = [r.spec_number for r in retrieval]
                eval_results.append(evaluate_one(sample, retrieved_specs))
            except Exception as e:
                logger.warning("  第 %d 条失败: %s", i + 1, e)

        from tests.eval.metrics import evaluate_batch
        report = evaluate_batch(eval_results)
        results[label] = {
            "recall_at_5": report.recall_at_5,
            "recall_at_10": report.recall_at_10,
            "recall_at_20": report.recall_at_20,
            "mrr": report.mrr,
            "ndcg_at_10": report.ndcg_at_10,
            "total": report.total,
        }
        logger.info(
            "  %s: R@5=%.4f R@10=%.4f MRR=%.4f NDCG@10=%.4f",
            label, report.recall_at_5, report.recall_at_10,
            report.mrr, report.ndcg_at_10,
        )
        store.disconnect()

    return results


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(description="BGE-M3 重索引 + 对比")
    parser.add_argument("--embed-only", action="store_true", help="仅嵌入 (不入库)")
    parser.add_argument("--eval-only", action="store_true", help="仅评测 (需两 collection 就绪)")
    parser.add_argument("--dry-run", action="store_true", help="预览 (不执行)")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评测")
    parser.add_argument("--no-mps", action="store_true", help="强制禁用 MPS (CPU 模式)")
    args = parser.parse_args()

    t_start = time.time()

    if args.eval_only:
        report = run_eval_comparison()
        print_summary({}, {}, report)
        return

    # ── 1. 读取现有 collection ──
    logger.info("=" * 60)
    logger.info("  BGE-M3 重索引流水线")
    logger.info("=" * 60)

    logger.info("\n── 1. 读取现有 collection (bge-large-v1.5) ──")
    source_store = MilvusStore(collection_name=BGE_V1_COLLECTION)
    rows = read_chunks_from_collection(source_store, "v1")

    if not rows:
        logger.error("无数据! bge-large-v1.5 嵌入可能尚未完成")
        return

    texts = [str(r.get("text", "")) for r in rows]
    chunks = build_chunk_objects(rows)
    source_store.disconnect()
    logger.info("总 chunks: %d, 总文本: ~%d 字符", len(texts), sum(len(t) for t in texts))

    if args.dry_run:
        logger.info("\n[DRY RUN] 将嵌入 %d texts, 入库 collection=%s", len(texts), BGE_M3_COLLECTION)
        return

    # ── 2. 快速基准 ──
    logger.info("\n── 2. 快速基准 (子集 64 texts, MPS) ──")
    quick_results = compare_models_on_subset(texts, 64)

    # ── 3. BGE-M3 全量嵌入 ──
    logger.info("\n── 3. BGE-M3 全量嵌入 (%d texts, MPS spawn) ──", len(texts))
    m3_embeddings, m3_metrics = benchmark_embed(texts, BGE_M3_MODEL, "BGE-M3-full")
    m3_metrics["source_embedding_batches"] = (
        len(texts) + 31
    ) // 32  # batch_size=32, rough estimate

    if args.embed_only:
        # 保存到文件
        np_path = PROJECT_ROOT / "data" / "vectors" / "bge_m3_embeddings.npy"
        np.save(np_path, m3_embeddings)
        logger.info("嵌入已保存: %s", np_path)
        print_summary(quick_results, m3_metrics, {})
        return

    # ── 4. 入库 BGE-M3 ──
    logger.info("\n── 4. 入库 BGE-M3 → %s ──", BGE_M3_COLLECTION)
    m3_store = MilvusStore(collection_name=BGE_M3_COLLECTION)
    inserted = insert_to_collection(chunks, m3_embeddings, m3_store, "m3")

    # 单独保存 BM25 (避免丢失)
    bm25_path = settings.vectors_dir / "bm25_index_BGE-M3.pkl"
    # BM25 已在 insert_to_collection 中构建并保存到默认路径，这里不额外操作
    m3_store.disconnect()

    m3_metrics["inserted_chunks"] = inserted

    # ── 5. 评测对比 ──
    eval_report = {}
    if not args.skip_eval:
        logger.info("\n── 5. 评测对比 ──")
        eval_report = run_eval_comparison()

    # ── 6. 汇总 ──
    print_summary(quick_results, m3_metrics, eval_report, time.time() - t_start)
    save_report(quick_results, m3_metrics, eval_report, time.time() - t_start)


def print_summary(
    quick_results: dict,
    m3_metrics: dict,
    eval_report: dict,
    total_elapsed: float = 0,
):
    """打印汇总报告."""
    print("\n" + "=" * 60)
    print("  BGE-M3 vs bge-large-en-v1.5 横向对比报告")
    print("=" * 60)

    # 速度
    if quick_results:
        v1_tps = quick_results.get("bge-large-v1.5", {}).get("texts_per_second", 0)
        m3_tps = quick_results.get("BGE-M3", {}).get("texts_per_second", 0)
        v1_mem = quick_results.get("bge-large-v1.5", {}).get("peak_memory_mb", 0)
        m3_mem = quick_results.get("BGE-M3", {}).get("peak_memory_mb", 0)
        print(f"\n  【速度】(MPS, 子集 64 texts)")
        print(f"    bge-large-v1.5:  {v1_tps:>8.0f} t/s,  peak {v1_mem:.0f} MB")
        print(f"    BGE-M3:          {m3_tps:>8.0f} t/s,  peak {m3_mem:.0f} MB")
        if v1_tps > 0:
            ratio = m3_tps / v1_tps
            tag = "更快 ⚡" if ratio > 1.05 else ("更慢 🐢" if ratio < 0.95 else "持平 ➡️")
            print(f"    速度比: {ratio:.2f}x  {tag}")

    # 全量
    if m3_metrics:
        tps = m3_metrics.get("texts_per_second", 0)
        total = m3_metrics.get("total_texts", 0)
        elapsed = m3_metrics.get("total_time_s", 0)
        nan = m3_metrics.get("nan_count", 0)
        zero = m3_metrics.get("zero_vector_count", 0)
        inserted = m3_metrics.get("inserted_chunks", 0)
        print(f"\n  【全量 BGE-M3】(MPS spawn)")
        print(f"    文本数: {total}")
        print(f"    耗时:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"    吞吐:   {tps:.0f} t/s")
        print(f"    维度:   {m3_metrics.get('embedding_dim', '?')}")
        print(f"    NaN:    {nan}  零向量: {zero}")
        if inserted:
            print(f"    入库:   {inserted} chunks")

    # 检索质量
    if eval_report:
        print(f"\n  【检索质量】(60 题 3GPP 测试集)")
        print(f"    {'指标':<20} {'bge-large-v1.5':<16} {'BGE-M3':<16} {'差异':<10}")
        print(f"    {'─' * 20} {'─' * 16} {'─' * 16} {'─' * 10}")
        for key, label in [
            ("recall_at_5", "Recall@5"),
            ("recall_at_10", "Recall@10"),
            ("recall_at_20", "Recall@20"),
            ("mrr", "MRR"),
            ("ndcg_at_10", "NDCG@10"),
        ]:
            v1_val = eval_report.get("bge-large-v1.5", {}).get(key, 0)
            m3_val = eval_report.get("BGE-M3", {}).get(key, 0)
            diff = m3_val - v1_val
            tag = "✅ +" if diff > 0.01 else ("❌ " if diff < -0.01 else "  ≈")
            print(f"    {label:<20} {v1_val:<16.4f} {m3_val:<16.4f} {tag}{diff:+.4f}")

    if total_elapsed > 0:
        print(f"\n  总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")


def save_report(
    quick_results: dict,
    m3_metrics: dict,
    eval_report: dict,
    elapsed: float,
):
    """保存 JSON 报告."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_s": elapsed,
        "quick_benchmark": quick_results,
        "m3_full_metrics": {k: v for k, v in m3_metrics.items()
                            if not isinstance(v, np.ndarray)},
        "eval_comparison": eval_report,
    }
    report_path = PROJECT_ROOT / "data" / "vectors" / "bge_m3_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("报告已保存: %s", report_path)


if __name__ == "__main__":
    main()
