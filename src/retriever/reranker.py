"""Cross-Encoder Reranker — 对混合检索候选做第二阶段精排.

BGE-Reranker-v2-m3 (Cross-Encoder) 将 (query, doc) 对送入完整 Transformer
cross-attention, 捕捉 Bi-Encoder 丢失的细粒度语义匹配. 典型使用场景:

    reranker = CrossEncoderReranker()
    refined = reranker.rerank(query, candidates, top_k=10)

架构: 可选启用, 与 BGE-M3 Bi-Encoder 互补而非替代.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.config import settings
from src.retriever.search import RetrievalResult

logger = logging.getLogger(__name__)

# ── 全局懒加载实例 ──
_reranker_instance: "CrossEncoderReranker | None" = None
_cross_encoder_model: Any = None


def get_reranker() -> "CrossEncoderReranker | None":
    """获取全局 reranker 实例 (懒加载, 仅当 settings.reranker_enabled 时)."""
    if not settings.reranker_enabled:
        return None
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class CrossEncoderReranker:
    """Cross-Encoder 重排序器.

    使用 BGE-Reranker-v2-m3 对候选文档进行 (query, doc) 对打分, 按相关性降序
    返回前 top_k 条. 首次调用时懒加载模型.

    Attributes:
        model: sentence_transformers.CrossEncoder 实例
        top_k: 重排序后返回的条数
        batch_size: 批量推理大小
    """

    def __init__(
        self,
        model_path: str | None = None,
        top_k: int | None = None,
        batch_size: int | None = None,
    ):
        self._model_path = model_path or settings.reranker_model
        self._top_k = top_k or settings.reranker_top_k
        self._model: Any = None
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            self._batch_size = 32  # HIGH_WATERMARK 已控制 MPS 内存, 无需缩小

    @property
    def model(self):
        """懒加载 CrossEncoder 模型."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = _resolve_reranker_device()
            # 如果配置的是 HuggingFace ID 但本地有同名目录, 优先本地
            model_id = self._model_path
            logger.info(
                "加载 Cross-Encoder Reranker: %s (device=%s, top_k=%d)",
                model_id, device, self._top_k,
            )
            self._model = CrossEncoder(
                model_id,
                device=device,
                trust_remote_code=True,  # BGE-Reranker 需要此选项
                max_length=1024,  # 模型训练用 1024, 默认 512 会截断长 chunk
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """对候选列表重排序, 返回 top_k 条.

        Args:
            query: 用户原始查询
            candidates: 混合检索候选 (建议 50-200 条)
            top_k: 返回条数, None 则用实例默认值

        Returns:
            按 Cross-Encoder 分数降序排列的 top_k 条结果
        """
        if not candidates:
            return []

        k = top_k or self._top_k
        # 仅当使用实例默认 top_k 时, 候选数不足才跳过推理 (性能优化);
        # 显式传入 top_k (如融合场景的全池打分) 必须执行推理以得到分数
        if top_k is None and len(candidates) <= k:
            return candidates

        # 构建 (query, doc) 对
        pairs = [(query, c.text) for c in candidates]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()
            else:
                scores = np.asarray(scores, dtype=np.float32)

            # MPS 每轮推理后释放 Metal 中间张量 (PyTorch ≥2.0 支持)
            self._mps_empty_cache()
        except Exception as e:
            logger.warning("Cross-Encoder 推理失败, 降级返回原始候选: %s", e)
            return candidates[:k]

        # 将新分数写入副本, 原地排序降级
        reranked: list[RetrievalResult] = []
        for result, score in zip(candidates, scores):
            result.score = float(score)
            reranked.append(result)

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载."""
        return self._model is not None

    def _mps_empty_cache(self) -> None:
        """MPS 设备上释放 Metal 中间张量缓存 (PyTorch ≥2.0).

        与 CUDA 不同, MPS 的 empty_cache 释放的是 Metal 命令缓冲区和
        中间分配, 不保证立即回收所有内存 (MPS 内部有自己的回收策略).
        但连续推理时主动调用可显著降低 wired 内存累积.
        """
        try:
            import torch
            if (
                hasattr(torch, "mps")
                and hasattr(torch.mps, "empty_cache")
                and torch.backends.mps.is_available()
            ):
                torch.mps.empty_cache()
        except Exception:
            pass  # 非关键路径, 静默降级


def _resolve_reranker_device() -> str:
    """解析 reranker 计算设备."""
    device = settings.reranker_device
    if device != "auto":
        return device

    # auto: 自动检测
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
