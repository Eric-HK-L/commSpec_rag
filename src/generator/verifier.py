"""答案验证 — 事实溯源校验，防止 LLM 幻觉."""

from __future__ import annotations

import logging
import re

from src.retriever.search import RetrievalResult

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """答案事实溯源验证器."""

    def verify(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> dict:
        """验证答案的事实基础."""
        warnings: list[str] = []

        # 检查 LLM 是否主动承认不确定性 (模糊匹配, 多了修饰词也能命中)
        uncertainty_keywords = [
            "无法确定", "未提及", "无法确认", "未找到相关",
            "cannot determine", "not mentioned", "cannot confirm",
        ]
        for kw in uncertainty_keywords:
            if kw.lower() in answer.lower():
                return {
                    "answer": answer,
                    "verified": True,
                    "warnings": [],
                    "coverage": 1.0,
                }

        # 检查引用的 TS 编号是否在检索结果中
        spec_refs = re.findall(r"TS\s*(\d{2}\.\d{3})", answer)
        source_specs = {s.spec_number for s in sources if s.spec_number}

        for ref in spec_refs:
            if ref not in source_specs:
                warnings.append(
                    f"\u26a0\ufe0f 答案引用了 TS {ref}，但该规范不在检索结果中，可能为幻觉"
                )

        # 文本重合度检查
        answer_lower = answer.lower()
        cited_count = 0
        # 按英文句尾分割 (句号/问号/感叹号后跟空格), 避免 TS 38.300 / V18.4.0 被错误切分
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
        total_sentences = len(sentences)

        for source in sources:
            if len(source.text) > 50:
                key_phrases = self._extract_key_phrases(source.text, min_length=20)
                for phrase in key_phrases:
                    if phrase.lower() in answer_lower:
                        cited_count += 1
                        break

        if total_sentences > 0:
            coverage = min(1.0, cited_count / max(total_sentences, 1))
        else:
            coverage = 1.0

        if coverage < 0.3 and len(sources) > 0:
            warnings.append(
                f"\u26a0\ufe0f 答案与检索结果文本重合度较低 ({coverage:.0%})，建议人工审核"
            )

        return {
            "answer": answer,
            "verified": len(warnings) == 0,
            "warnings": warnings,
            "coverage": coverage,
        }

    @staticmethod
    def _extract_key_phrases(text: str, min_length: int = 20) -> list[str]:
        """从文本中提取关键短语."""
        phrases = []
        parts = re.split(r"[.\n]", text)
        for part in parts:
            stripped = part.strip()
            if len(stripped) >= min_length:
                phrases.append(stripped)
        return phrases[:5]
