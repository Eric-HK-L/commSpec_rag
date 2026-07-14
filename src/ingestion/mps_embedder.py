"""Apple Silicon MPS 批量嵌入器 — spawn 单 worker + 生命周期回收.

=== MPS 核心约束 ===

1. Apple MPS 不支持多进程同时使用 GPU:
   多进程同时调用 [MTLCommandBuffer waitUntilCompleted] 会触发 Metal
   GPU 调度器死锁 — 所有 worker 永久阻塞在 __psynch_cvwait。
   这是 Apple Metal 框架层面的限制, 与 Python multiprocessing 无关。
   → workers 必须 = 1。

2. MPS 无 empty_cache() (torch.mps.empty_cache() 不存在):
   统一内存架构下 GPU 缓存无法主动释放, 单进程长时间嵌入导致
   内存单向增长直至 OOM。
   → 单 worker 用 chunks_per_worker 控制生命周期, 定期退出让 OS 回收。

=== 本模块方案 ===

  - spawn 创建全新 OS 进程 (fork 对 Metal 是 undefined behavior)
  - 父进程永不 import torch, 零 Metal 框架状态
  - 单 worker 独立加载模型 (local_files_only) → 嵌入 N 批 → 退出
  - OS 回收退出 worker 的全部 GPU/统一内存
  - ProcessPoolExecutor (非 Pool) 管理 worker 生命周期
    (Pool 的 SimpleQueue/os.pipe 在 macOS spawn 下 FD 泄漏 → 死锁)

=== 性能 (M4 Max, 108K chunks, BGE-M3, workers=1) ===

  - 模型加载: ~1.5s (首次) + 每 chunks_per_worker 批后回收 ~1.5s
  - 嵌入计算: 3404 批 × ~0.03s/batch ≈ 102s (GPU 内部并行)
  - 总计: ~110s ≈ 2 分钟 (vs CPU 3.8h, ~120× 加速)

=== 曾尝试但不工作的方案 (备忘) ===

  - workers>1: MPS 多进程 GPU 死锁 (Metal [MTLCommandBuffer waitUntilCompleted])
  - multiprocessing.Pool: SimpleQueue/os.pipe FD 泄漏到 spawn 子进程
  - multiprocessing.Lock: 依赖 FD 继承, spawn 不继承 FD
  - fork 启动: Apple 文档 undefined behavior, SIGABRT 风险
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
import time
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# ── Worker 全局 (spawn 子进程中设置) ──
_WORKER_MODEL: Any = None
_INIT_LOCK_PATH: str = ""  # 文件锁路径 (跨 spawn 安全: 字符串, 非 FD)


def _worker_init(model_name: str, lock_path: str = ""):
    """Worker 初始化: 文件锁串行加载 BGE 模型 + MPS JIT 预热.

    multiprocessing.Lock 依赖 FD 继承, spawn 不继承 FD → unpickle 后
    锁失效 → 死锁。改用 fcntl.flock 文件锁, 路径是纯字符串, 跨 spawn 安全。
    """
    global _WORKER_MODEL, _INIT_LOCK_PATH
    _INIT_LOCK_PATH = lock_path
    try:
        from sentence_transformers import SentenceTransformer
        pid = os.getpid()

        if lock_path:
            import fcntl
            lock_file = open(lock_path, "w")
            logger.debug("Worker %s: 等待文件锁 %s...", pid, lock_path)
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                logger.debug("Worker %s: 获得锁, 加载 %s → mps", pid, model_name)
                _WORKER_MODEL = SentenceTransformer(
                    model_name, device="mps", local_files_only=True,
                )
                _WORKER_MODEL.encode(
                    ["warmup"], normalize_embeddings=True, show_progress_bar=False,
                )
                logger.debug("Worker %s: 初始化完成", pid)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
        else:
            _WORKER_MODEL = SentenceTransformer(
                model_name, device="mps", local_files_only=True,
            )
    except Exception as e:
        logger.error("Worker %s 初始化失败: %s", os.getpid(), e)
        raise


def _worker_embed(texts: list[str]) -> np.ndarray:
    """Worker 任务: 嵌入一批文本 → float32 数组.

    Worker 在处理 maxtasksperchild 个任务后自动退出 → OS 回收 MPS 内存.
    """
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker 模型未初始化")
    embeddings = _WORKER_MODEL.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


# ── MPS Chunked Embedder ──

class MPSChunkedEmbedder:
    """MPS 批量嵌入 — spawn 单 worker + 生命周期回收.

    约束: Apple MPS 不支持多进程 GPU → workers 必须 = 1。
    GPU 内部已高度并行, 单进程 T4≈50 t/s, M4 Max≈1000 t/s, 无需多 worker。

    父进程纯调度, 永不 import torch (零 Metal 风险).
    子进程独立 MPS → 嵌 N 批 → 退出 → OS 回收.

    使用:
        embedder = MPSChunkedEmbedder()
        embeddings = embedder.embed_raw(texts, batch_size=32)
    """

    # workers 硬编码为 1: Apple MPS 不支持多进程 GPU,
    # CUDA/CPU 场景应用 BatchEmbedder (embedder.py) 而非本类
    _WORKERS: int = 1

    def __init__(
        self,
        model_name: str = "",
        chunks_per_worker: int = 200,
    ):
        self._model_name = model_name or settings.local_embedding_model
        self._chunks_per_worker = max(10, chunks_per_worker)
        self._setup_spawn()
        # 文件锁路径: 串行化 worker 的 MPS 初始化 (纯字符串, 跨 spawn 安全)
        self._ctx = mp.get_context("spawn")
        self._lock_path = os.path.join(tempfile.gettempdir(), "mps_init.lock")

    @staticmethod
    def _setup_spawn():
        """强制使用 spawn — fork 对 Metal 不安全."""
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        logger.info("MPS embedder: spawn (fork 对 Metal 不安全)")

    @staticmethod
    def is_mps_available() -> bool:
        """检测 MPS 可用性 (仅在子进程 import torch)."""
        try:
            import torch
            return (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except ImportError:
            return False

    def embed_raw(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """纯嵌入计算 — ProcessPoolExecutor (非 Pool, macOS spawn 兼容) + worker 回收.

        Worker 回收策略: 将 batches 按 chunks_per_worker × num_workers
        切分为 mega-chunks, 每个 mega-chunk 创建新的 executor →
        executor 退出 → OS 回收该批次全部 MPS 内存。

        为什么 ProcessPoolExecutor 而非 Pool:
          Pool 内部 SimpleQueue 底层是 os.pipe(), macOS spawn 下
          pipe FD 被多 worker 继承 → 通信混乱 → imap 死锁。
          ProcessPoolExecutor 用 multiprocessing.Queue (Pipe + feeder 线程),
          spawn 安全, 与裸 Process 同等可靠。
        """
        if not texts:
            return np.empty((0, settings.embedding_dimension), dtype=np.float32)

        total = len(texts)
        mini_batches = [
            texts[i : i + batch_size] for i in range(0, total, batch_size)
        ]
        mega_size = self._chunks_per_worker * self._WORKERS
        mega_chunks = [
            mini_batches[i : i + mega_size]
            for i in range(0, len(mini_batches), mega_size)
        ]
        logger.info(
            "MPS spawn: %d texts → %d batches → %d mega-chunks (workers=%d, 回收/=%d批)",
            total, len(mini_batches), len(mega_chunks),
            self._WORKERS, mega_size,
        )

        from concurrent.futures import ProcessPoolExecutor

        t_start = time.time()
        all_results: list[np.ndarray] = []
        batch_offset = 0

        for mc_idx, mega_chunk in enumerate(mega_chunks):
            logger.debug("Mega-chunk %d/%d: %d batches", mc_idx + 1, len(mega_chunks), len(mega_chunk))

            with ProcessPoolExecutor(
                max_workers=self._WORKERS,
                mp_context=self._ctx,
                initializer=_worker_init,
                initargs=(self._model_name, self._lock_path),
            ) as executor:
                for emb in executor.map(_worker_embed, mega_chunk, chunksize=1):
                    all_results.append(emb)
                    batch_offset += 1
                    if batch_offset % 200 == 0 or batch_offset == len(mini_batches):
                        elapsed = time.time() - t_start
                        done = min(batch_offset * batch_size, total)
                        logger.info(
                            "  进度: %d/%d batches (%d/%d texts, %.0f t/s)",
                            batch_offset, len(mini_batches), done, total,
                            done / elapsed if elapsed > 0 else 0,
                        )

        elapsed = time.time() - t_start
        logger.info(
            "MPS 完成: %d texts, %.1fs (%.0f t/s), GPU 内存已随 executor 退出回收",
            total, elapsed, total / elapsed if elapsed > 0 else 0,
        )
        return np.concatenate(all_results, axis=0).astype(np.float32)
