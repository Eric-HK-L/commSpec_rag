"""LLM 客户端 — OpenAI 兼容 API 统一封装."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

# ── 本地嵌入模型（懒加载，避免启动时必须 torch）──
_local_embed_model: Any = None


def _get_local_embed_model():
    """懒加载 sentence-transformers 模型，自动选择最优设备."""
    global _local_embed_model
    if _local_embed_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = getattr(settings, 'local_embedding_model', 'BAAI/bge-m3')
        device = settings.resolved_embedding_device
        logger.info("加载本地嵌入模型: %s (device=%s)", model_name, device)
        _local_embed_model = SentenceTransformer(
            model_name, device=device,
            local_files_only=True,  # 禁止网络访问，加速加载
        )
    return _local_embed_model


class LLMClient:
    """OpenAI 兼容 LLM API 客户端.

    base_url + api_key + model 三参数覆盖:
    - 公司自建 API / OpenAI / Azure / Ollama
    """

    def __init__(self):
        self._client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.llm_model

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发送聊天请求."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM API 调用失败: %s", e)
            raise

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """流式聊天 — 逐段生成文本 (生成器).

        SSE 问答的核心: 边生成边推送, 用户无需等待完整回答.
        """
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("LLM 流式 API 调用失败: %s", e)
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本嵌入 — 根据 EMBEDDING_PROVIDER 使用本地或云端模型."""
        # 本地模式：直接使用 sentence-transformers
        if settings.embedding_provider == "local":
            return self._local_embed(texts)

        # 云端 API 优先，失败则降级本地
        try:
            response = self._client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.debug("云端嵌入失败，尝试本地模型: %s", e)

        return self._local_embed(texts)

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        """本地嵌入模型."""
        try:
            model = _get_local_embed_model()
            embeddings = model.encode(texts, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error("嵌入生成失败（本地模型不可用）: %s", e)
            raise

    @property
    def model(self) -> str:
        return self._model
