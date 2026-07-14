#!/usr/bin/env python3
"""MPS 嵌入子进程 — subprocess 隔离 + 单进程 MPS 直接编码.

设计要点:
1. subprocess 隔离: bulk_ingest.py 已 import pymilvus (gRPC),
   本文件独立 __main__, 干净无 gRPC 依赖。

2. 单进程 MPS 直接编码: workers=1 时 ProcessPoolExecutor 不仅无益,
   每个 batch 的 pickle IPC 开销远超 GPU 计算。3404 轮序列化/反序列化
   是上次 20 分钟超时的根因。本文件直接在主进程加载模型, 逐批编码,
   零 IPC 开销。

3. 生命周期回收: 每 chunks_per_worker 批后卸载+重载模型,
   OS 回收 MPS 缓存, 防止内存单向增长。

用法:
  python scripts/_mps_embed_subprocess.py <texts.pkl> <embeddings.npy>
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mps_subprocess")

# ── MPS 内存水位线 (必须在 import torch 之前设置) ──
# 默认 HIGH=1.0 允许 MPS 占用 100% 系统内存, 导致 wired 爆炸/OOM
# 经多轮 benchmark: batch_size=4 + HIGH=0.5 最优 (15.9 t/s, wired ~8GB)
# 根因: BGE-M3 注意力 O(n²), 大批次 padding 惩罚远超 GPU 并行收益
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")


def main():
    parser = argparse.ArgumentParser(description="MPS 嵌入子进程")
    parser.add_argument("input_file", help="pickle 文件")
    parser.add_argument("output_file", help="输出 .npy")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunks-per-worker", type=int, default=500)
    args = parser.parse_args()

    model_name = args.model or settings.local_embedding_model
    batch_size = args.batch_size
    recycle_every = args.chunks_per_worker  # 每 N 批后重载模型回收 MPS 缓存

    logger.info("读取文本: %s", args.input_file)
    with open(args.input_file, "rb") as f:
        texts = pickle.load(f)
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    logger.info("共 %d 条文本 → %d batches (batch_size=%d)", total, total_batches, batch_size)

    # ── 单进程 MPS 直接编码 ──
    from sentence_transformers import SentenceTransformer
    import torch

    all_embeddings: list[np.ndarray] = []
    t_start = time.time()

    model = None
    for batch_idx in range(total_batches):
        # 每 recycle_every 批或首次: (重)加载模型
        if batch_idx % recycle_every == 0:
            if model is not None:
                del model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
                logger.debug("模型已卸载, 回收 MPS 缓存 (batch %d/%d)", batch_idx, total_batches)
            t_load = time.time()
            device = args.device or settings.resolved_embedding_device
            model = SentenceTransformer(model_name, device=device, local_files_only=True)
            t_load = time.time() - t_load
            if batch_idx == 0:
                logger.info("模型加载: %s → %s (%.1fs)", model_name, device, t_load)
            else:
                logger.debug("模型重载: %.1fs", t_load)

        # 编码一批
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = texts[start:end]
        with torch.no_grad():
            emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
        all_embeddings.append(np.array(emb, dtype=np.float32))
        del emb

        # 每批次后主动释放 MPS 缓存 (CPU 侧触发, 防止 wired 内存单向累积)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # 每 10 批加一次 GC + sync（平衡开销）
        if batch_idx % 10 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()

        # 进度
        if (batch_idx + 1) % 200 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t_start
            done = end
            logger.info(
                "进度: %d/%d batches (%d/%d texts, %.0f t/s)",
                batch_idx + 1, total_batches, done, total,
                done / elapsed if elapsed > 0 else 0,
            )

    elapsed = time.time() - t_start
    result = np.concatenate(all_embeddings, axis=0)
    logger.info(
        "嵌入完成: %d vectors, dim=%d, %.1fs (%.0f t/s)",
        total, result.shape[1], elapsed, total / elapsed if elapsed > 0 else 0,
    )

    np.save(args.output_file, result)
    logger.info("已保存: %s (%.1f MB)", args.output_file, os.path.getsize(args.output_file) / 1e6)
    print(f"OK {len(result)}", flush=True)


if __name__ == "__main__":
    main()
