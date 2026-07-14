"""NN Router — 神经网络查询路由，预测最相关 3GPP Series.

注意: 依赖 torch，仅在 ENABLE_NN_ROUTER=true 且模型文件存在时才加载.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

SERIES_LABELS = np.arange(21, 39)
SERIES_DESCRIPTIONS = [
    "Requirements (21 series): overarching requirements for UMTS and later cellular standards.",
    "Service aspects stage 1 (22 series): initial specifications for services.",
    "Technical realization stage 2 (23 series): architectural and functional framework.",
    "Signalling protocols stage 3 - UE to network (24 series).",
    "Radio aspects (25 series): radio transmission technologies.",
    "CODECs (26 series): voice, audio, and video codecs.",
    "Data (27 series): data services and capabilities.",
    "Signalling protocols stage 3 - RSS-CN and OAM&P and Charging (28 series).",
    "Signalling protocols stage 3 - intra-fixed-network (29 series).",
    "Programme management (30 series).",
    "Subscriber Identity Module SIM/USIM, IC Cards (31 series).",
    "OAM&P and Charging (32 series).",
    "Security aspects (33 series).",
    "UE and (U)SIM test specifications (34 series).",
    "Security algorithms (35 series).",
    "LTE, LTE-Advanced, LTE-Advanced Pro radio technology (36 series).",
    "Multiple radio access technology aspects (37 series).",
    "Radio technology beyond LTE (38 series).",
]


class NNRouterModel:
    """NN Router 模型 — 使用 torch 时延迟初始化.

    架构: 1024-dim embedding + 18-dim series similarity -> 18-class output.
    支持 INT8 动态量化 (模型 ~5MB → ~1.2MB, 推理 ~20ms → ~5ms).
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        quantize: bool = True,
    ):
        self._model = None
        self._device = device
        self._model_path = model_path
        self._loaded = False
        self._quantize = quantize
        self._quantized = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _NNRouterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1_1 = nn.Linear(1024, 768)
                self.layer1_2 = nn.Linear(768, 512)
                self.layer1_3 = nn.Linear(512, 256)
                self.dropout1 = nn.Dropout(0.2)
                self.layer2_1 = nn.Linear(18, 128)
                self.layer2_2 = nn.Linear(128, 256)
                self.dropout2 = nn.Dropout(0.05)
                self.batchnorm1 = nn.BatchNorm1d(256)
                self.batchnorm2 = nn.BatchNorm1d(256)
                self.alfa = nn.Parameter(torch.ones(1), requires_grad=True)
                self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
                self.output_layer1 = nn.Linear(256, 128)
                self.output_layer2 = nn.Linear(128, 18)
                self.leaky_relu = nn.LeakyReLU(0.01)

            def forward(self, input_1, input_2):
                x1 = F.relu(self.layer1_1(input_1))
                x1 = self.dropout1(x1)
                x1 = F.relu(self.layer1_2(x1))
                x1 = self.dropout1(x1)
                x1 = F.relu(self.layer1_3(x1))
                x1 = self.batchnorm1(x1)
                x2 = F.relu(self.layer2_1(input_2))
                x2 = self.dropout2(x2)
                x2 = F.relu(self.layer2_2(x2))
                x2 = self.batchnorm2(x2)
                combined = self.alfa * x1 + self.beta * x2
                output = self.output_layer1(self.leaky_relu(combined))
                output = self.output_layer2(self.leaky_relu(output))
                return output

        self._model = _NNRouterModel().to(torch.device(self._device))
        self._model.eval()

        if self._model_path and os.path.exists(self._model_path):
            state = torch.load(self._model_path, map_location=self._device)
            self._model.load_state_dict(state)
            logger.info("NN Router 模型加载: %s", self._model_path)

        # INT8 动态量化 — 仅量化 nn.Linear 层, 保持精度
        # Apple Silicon (MPS) 不支持量化, 仅在 CPU + fbgemm/qnnpack 时启用
        if self._quantize and self._device == "cpu":
            try:
                # ARM (Apple Silicon) → qnnpack; x86 → fbgemm
                if hasattr(torch.backends, 'quantized'):
                    import platform
                    if platform.machine() == "arm64":
                        torch.backends.quantized.engine = "qnnpack"
                    else:
                        torch.backends.quantized.engine = "fbgemm"

                self._model = torch.quantization.quantize_dynamic(
                    self._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                self._quantized = True
                # 统计量化效果
                size_mb = sum(p.numel() * p.element_size() for p in self._model.parameters()) / 1024 / 1024
                logger.info(
                    "NN Router INT8 量化完成: %.1f MB (qint8)",
                    size_mb,
                )
            except Exception as e:
                logger.debug("INT8 量化跳过 (可能无硬件加速): %s", e)

        # 预热 — 避免首次推理延迟
        self._warmup()

        self._loaded = True

    def _warmup(self):
        """预热模型: 用零向量跑一次推理, 触发 JIT/内核编译."""
        try:
            import torch
            dummy_emb = torch.zeros(1, 1024)
            dummy_sim = torch.zeros(1, 18)
            with torch.no_grad():
                _ = self._model(dummy_emb, dummy_sim)
            logger.debug("NN Router 预热完成")
        except Exception as e:
            logger.debug("NN Router 预热跳过: %s", e)

    def predict(self, query_embedding: np.ndarray, top_k: int = 5) -> list[int]:
        """预测最相关 Series."""
        self._ensure_loaded()
        import torch

        emb = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
        sim = torch.zeros(1, 18)
        with torch.inference_mode():
            outputs = self._model(emb, sim)
            _, top_indices = outputs.topk(min(top_k, 18), dim=1)
            predicted = SERIES_LABELS[top_indices.numpy()]
        return list(predicted[0])

    def predict_series(self, query_embedding: np.ndarray, top_k: int = 5) -> list[int]:
        """predict 的别名."""
        return self.predict(query_embedding, top_k)


class NNRouter:
    """NN Router 封装 — 预测查询最相关的 3GPP Series."""

    def __init__(self, model_path: str | None = None, device: str = "cpu", quantize: bool = True):
        self._inner = NNRouterModel(model_path=model_path, device=device, quantize=quantize)

    def predict_top_series(
        self, query: str, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[int]:
        """预测查询最相关的 Series 编号."""
        try:
            return self._inner.predict(query_embedding, top_k=top_k)
        except Exception as e:
            logger.error("NN Router 预测失败: %s", e)
            return []
