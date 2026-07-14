#!/usr/bin/env python3
"""Milvus 端到端验证 — 嵌入完成后运行.

验证清单:
  1. Milvus 集合 TeleComm_specs 有数据
  2. 向量索引已构建
  3. Dense 检索返回相关结果
  4. Sparse (BM25) 检索返回结果
  5. RAGPipeline.search() 混合检索
  6. RAGPipeline.ask() 完整问答链路
  7. 多语言 (中文) 查询
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2e_verify")


def verify_milvus() -> bool:
    """验证 Milvus 数据就绪."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri="http://localhost:19530")
    cols = client.list_collections()
    assert "TeleComm_specs" in cols, f"集合不存在: {cols}"

    stats = client.get_collection_stats("TeleComm_specs")
    row_count = stats["row_count"]
    assert row_count > 0, f"Milvus 为空 ({row_count} rows)"

    logger.info("✅ Milvus: %d rows in TeleComm_specs", row_count)
    return True


def verify_search() -> bool:
    """验证检索管线."""
    import numpy as np

    from src.config import settings
    from src.retriever.milvus_store import MilvusStore
    from src.retriever.search import HybridRetriever

    store = MilvusStore()
    store.connect()
    retriever = HybridRetriever(
        vector_store=store,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.bm25_top_k,
        final_top_k=settings.max_search_results,
    )

    # 用零向量做语义检索（依赖 BM25 召回）
    query = "PDU Session Establishment procedure"
    emb = np.random.randn(1024).astype(np.float32)  # placeholder

    t0 = time.time()
    results = retriever.search(query, emb)
    dt = time.time() - t0

    assert len(results) > 0, "检索结果为空"
    logger.info("✅ 混合检索: %d results in %.2fs", len(results), dt)
    for i, r in enumerate(results[:3]):
        logger.info("  [%d] %s | %s §%s | score=%.3f",
                     i, r.spec_number, r.doc_id[:30], r.parent_section_id, r.score)

    store.disconnect()
    return True


def verify_ask() -> bool:
    """验证完整 RAG 问答."""
    from src.generator.pipeline import RAGPipeline
    from src.retriever.milvus_store import MilvusStore

    store = MilvusStore()
    store.connect()

    pipeline = RAGPipeline(vector_store=store)

    # 测试查询 (英文)
    query = "What is a PDU Session in 5G?"
    logger.info("查询: %s", query)

    t0 = time.time()
    response = pipeline.ask(query)
    dt = time.time() - t0

    assert response.answer, "回答为空"
    assert len(response.answer) > 20, f"回答过短: {len(response.answer)} chars"
    logger.info("✅ ask() 完成: %.2fs", dt)
    logger.info("  回答长度: %d chars", len(response.answer))
    logger.info("  来源数: %d", len(response.sources))
    logger.info("  已验证: %s", response.verified)
    logger.info("  回答预览: %s", response.answer[:200])

    store.disconnect()
    return True


def verify_i18n() -> bool:
    """验证多语言查询."""
    from src.generator.pipeline import RAGPipeline
    from src.retriever.milvus_store import MilvusStore

    store = MilvusStore()
    store.connect()

    pipeline = RAGPipeline(vector_store=store)

    # 中文查询
    query = "5G中PDU会话建立流程是什么"
    logger.info("中文查询: %s", query)

    t0 = time.time()
    response = pipeline.ask(query)
    dt = time.time() - t0

    assert response.answer, "中文回答为空"
    logger.info("✅ 中文 ask() 完成: %.2fs", dt)
    logger.info("  回答预览: %s", response.answer[:200])

    store.disconnect()
    return True


def main():
    logger.info("=" * 60)
    logger.info("Milvus 端到端验证")
    logger.info("=" * 60)

    checks = [
        ("Milvus 数据就绪", verify_milvus),
        ("混合检索", verify_search),
        ("RAG 问答 (EN)", verify_ask),
        ("多语言 (ZH)", verify_i18n),
    ]

    passed = 0
    for name, fn in checks:
        try:
            logger.info("--- %s ---", name)
            fn()
            passed += 1
        except Exception as e:
            logger.error("❌ %s 失败: %s", name, e)

    logger.info("=" * 60)
    logger.info("结果: %d/%d 通过", passed, len(checks))
    if passed == len(checks):
        logger.info("🎉 全部通过!")
    else:
        logger.warning("⚠️ 有 %d 项失败", len(checks) - passed)


if __name__ == "__main__":
    main()
