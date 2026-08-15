#!/usr/bin/env python3
"""按 embedding_text_mode 重新嵌入全量 chunk — 保留原 id, 写入新 collection.

用于 embedding_text A/B: 只改 Dense 向量, text/元数据/BM25/xref 图全部不变,
从而隔离"层级路径入向量"这一单一变量的影响。

用法:
  EMBEDDING_TEXT_MODE=path_text python scripts/reembed_embedding_text.py \
      --target TeleComm_specs_path_text

流程:
  1. 读源 collection (settings.milvus_collection_name) 全部 chunk (id + 全字段)
  2. 用 embedding_text(chunk) (受 EMBEDDING_TEXT_MODE 控制) 重嵌 (MPS spawn)
  3. 新建 <target> collection (auto_id=False, 保留原 id)
  4. 入库; BM25 无需重建 (text 未变, 与源 collection 共享 doc_key 空间)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.embedder import embedding_text  # noqa: E402
from src.ingestion.mps_embedder import MPSChunkedEmbedder  # noqa: E402
from src.retriever.milvus_store import (  # noqa: E402
    VARCHAR_MAX,
    MilvusStore,
    _safe_truncate_bytes,
)
from src.retriever.vector_store import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reembed")

# 与 milvus_store.create_collection 相同的字段, 仅 id 改为 auto_id=False (保留原 id)
_FIELDS = [
    ("id", "INT64", True),  # (name, dtype, is_primary)
    ("text", "VARCHAR", VARCHAR_MAX),
    ("dense_vector", "FLOAT_VECTOR", settings.embedding_dimension),
    ("doc_id", "VARCHAR", 256),
    ("series", "INT64", None),
    ("spec_number", "VARCHAR", 64),
    ("release", "VARCHAR", 32),
    ("version", "VARCHAR", 32),
    ("parent_section_id", "VARCHAR", 256),
    ("parent_title", "VARCHAR", 4096),
    ("chunk_index", "INT64", None),
    ("section_number", "VARCHAR", 64),
    ("section_title", "VARCHAR", 512),
    ("section_path", "VARCHAR", 4096),
    ("doc_type", "VARCHAR", 32),
    ("content_type", "VARCHAR", 32),
    ("spec_role", "VARCHAR", 32),
    ("topic_domain", "VARCHAR", 32),
    ("parent_chunk_id", "INT64", None),
    ("parent_text", "VARCHAR", 4096),
]


def read_source_chunks(src: MilvusStore) -> list[dict]:
    """分页读取源 collection 全部 chunk (Milvus query 单次 limit ≤ 16384)."""
    src.connect()
    src._ensure_connected()
    col = src._collection
    output_fields = [f[0] for f in _FIELDS if f[0] != "dense_vector"]
    rows: list[dict] = []
    last_id = -1
    batch = 8000
    while True:
        batch_rows = col.query(
            expr=f"id > {last_id}", output_fields=output_fields, limit=batch,
        )
        if not batch_rows:
            break
        rows.extend(batch_rows)
        last_id = batch_rows[-1]["id"]
        if len(batch_rows) < batch:
            break
        logger.info("  已读 %d chunks (last_id=%d)", len(rows), last_id)
    logger.info("读取源 collection %s: %d chunks", src._collection_name, len(rows))
    return rows


def build_embedding_texts(rows: list[dict]) -> list[str]:
    texts: list[str] = []
    for r in rows:
        c = Chunk(
            text=str(r.get("text", "")),
            doc_id=str(r.get("doc_id", "")),
            series=int(r.get("series", 0)),
            spec_number=str(r.get("spec_number", "")),
            release=str(r.get("release", "")),
            version=str(r.get("version", "")),
            parent_section_id=str(r.get("parent_section_id", "")),
            parent_title=str(r.get("parent_title", "")),
            chunk_index=int(r.get("chunk_index", 0)),
            section_number=str(r.get("section_number", "")),
            section_title=str(r.get("section_title", "")),
            section_path=str(r.get("section_path", "")),
            doc_type=str(r.get("doc_type", "3gpp")),
            content_type=str(r.get("content_type", "")),
            spec_role=str(r.get("spec_role", "")),
            topic_domain=str(r.get("topic_domain", "")),
            parent_chunk_id=int(r.get("parent_chunk_id", 0)),
            parent_text=str(r.get("parent_text", "")),
        )
        texts.append(embedding_text(c))
    return texts


def create_target_collection(target: str) -> None:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

    if utility.has_collection(target):
        utility.drop_collection(target)
        logger.info("已删除旧目标 collection: %s", target)

    fields = []
    for name, dtype, primary in _FIELDS:
        if name == "id":
            fields.append(FieldSchema(name=name, dtype=DataType.INT64, is_primary=True))
        elif dtype == "VARCHAR":
            fields.append(FieldSchema(name=name, dtype=DataType.VARCHAR, max_length=primary))
        elif dtype == "FLOAT_VECTOR":
            fields.append(FieldSchema(name=name, dtype=DataType.FLOAT_VECTOR, dim=primary))
        elif dtype == "INT64":
            fields.append(FieldSchema(name=name, dtype=DataType.INT64))
    schema = CollectionSchema(fields, description="re-embed A/B (auto_id=False)")
    col = Collection(target, schema)
    col.create_index(
        field_name="dense_vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        },
    )
    col.load()
    logger.info("目标 collection 创建并加载: %s", target)


def insert_with_ids(target: str, rows: list[dict], embeddings: np.ndarray) -> int:
    from pymilvus import Collection

    col = Collection(target)
    batch = 1000
    total = len(rows)
    inserted = 0
    for start in range(0, total, batch):
        chunk_rows = rows[start : start + batch]
        data: list[list] = [[] for _ in _FIELDS]
        for i, r in enumerate(chunk_rows):
            emb = embeddings[start + i]
            for j, (name, dtype, _) in enumerate(_FIELDS):
                if name == "id":
                    data[j].append(int(r["id"]))
                elif name == "text":
                    data[j].append(_safe_truncate_bytes(str(r.get("text", "")), VARCHAR_MAX))
                elif name == "dense_vector":
                    data[j].append(np.asarray(emb, dtype=np.float32).tolist())
                elif dtype == "VARCHAR":
                    data[j].append(_safe_truncate_bytes(str(r.get(name, "")), _max_len(name)))
                else:  # INT64
                    data[j].append(int(r.get(name, 0)))
        col.insert(data)
        inserted += len(chunk_rows)
        if (start // batch + 1) % 5 == 0:
            logger.info("  入库进度: %d/%d", inserted, total)
    col.flush()
    return inserted


def _max_len(name: str) -> int:
    for n, dtype, v in _FIELDS:
        if n == name:
            return v
    return 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    mode = settings.embedding_text_mode
    if mode == "text":
        logger.warning("EMBEDDING_TEXT_MODE=text 与现状相同, 请设 EMBEDDING_TEXT_MODE=path_text")
        return

    src = MilvusStore(collection_name=settings.milvus_collection_name)
    rows = read_source_chunks(src)
    if not rows:
        logger.error("源 collection 无数据")
        return

    texts = build_embedding_texts(rows)
    logger.info("embedding_text_mode=%s, 样例: %.80s", mode, texts[0])

    t0 = time.time()
    embedder = MPSChunkedEmbedder(model_name=settings.local_embedding_model)
    embeddings = embedder.embed_raw(texts, batch_size=args.batch_size)
    logger.info("重嵌入完成: %d chunks, %.1fs", len(texts), time.time() - t0)

    create_target_collection(args.target)
    n = insert_with_ids(args.target, rows, embeddings)
    logger.info("入库完成: %d/%d → %s", n, len(rows), args.target)
    logger.info(
        "验证: MILVUS_COLLECTION_NAME=%s 跑 tests/eval/run_eval.py 即可对比 (BM25 复用, 无需重建)",
        args.target,
    )


if __name__ == "__main__":
    main()
