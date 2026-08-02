"""多语言支持 — 语言检测 + 查询翻译(→EN) + 回答回译(→用户语言).

支持语言: 中文(zh)、英文(en)、韩文(ko).
策略: 非英文查询 → LLM 翻译为英文 → 英文检索 → 回答回译为原文语言.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generator.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ── 语言标签映射 ──
_LANG_NAMES = {
    "zh": "Simplified Chinese",
    "ko": "Korean",
    "en": "English",
}

# ── 字符集检测 ──
_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")  # 中文
_RE_KOREAN = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")  # 韩文
_RE_JAPANESE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")  # 日文假名 (排除用)


def detect_language(text: str) -> str:
    """检测查询语言. 返回 'zh' | 'en' | 'ko' | 'unknown'.

    策略: 统计中文/韩文字符数量，最多者胜出.
    """
    cjk_count = len(_RE_CJK.findall(text))
    ko_count = len(_RE_KOREAN.findall(text))

    if cjk_count > ko_count and cjk_count > 0:
        return "zh"
    if ko_count > cjk_count and ko_count > 0:
        return "ko"
    # 没有 CJK/Korean 字符 → 英文
    return "en"


def needs_translation(lang: str) -> bool:
    """是否需要翻译为英文."""
    return lang != "en"


# ── 翻译提示词 ──

_TRANSLATE_TO_EN_SYSTEM = """You are a professional telecom standards translator (3GPP / O-RAN).
Translate the user's query into accurate, technical English.
Rules:
1. Preserve all specification numbers (TS xx.xxx, TR xx.xxx)
2. Preserve all technical abbreviations (NR, LTE, PUSCH, DMRS, QoS, RRC, MAC, etc.)
3. Use precise telecom standards terminology
4. Output ONLY the English translation, no explanations"""

_TRANSLATE_FROM_EN_SYSTEM = """You are a professional telecom standards translator (3GPP / O-RAN).
Translate the following English answer into {target_lang_name}.
Rules:
1. Preserve all specification numbers (TS xx.xxx, TR xx.xxx)
2. Preserve all technical abbreviations (NR, LTE, PUSCH, DMRS, QoS, RRC, MAC, etc.)
3. Use natural, fluent {target_lang_name}
4. Output ONLY the translated answer, no explanations"""


def translate_to_english(
    text: str,
    source_lang: str,
    llm: LLMClient,
) -> str:
    """将非英文查询翻译为英文.

    Args:
        text: 原始查询文本.
        source_lang: 源语言代码 ('zh' | 'ko').
        llm: LLM 客户端.

    Returns:
        英文翻译文本. 失败时返回原文.
    """
    if not needs_translation(source_lang):
        return text

    lang_name = _LANG_NAMES.get(source_lang, source_lang)
    messages = [
        {"role": "system", "content": _TRANSLATE_TO_EN_SYSTEM},
        {"role": "user", "content": f"Translate this {lang_name} query to English:\n\n{text}"},
    ]

    try:
        result = llm.chat(messages, temperature=0.0, max_tokens=512)
        translated = result.strip()
        if translated and len(translated) > 3:
            logger.info("查询翻译: %s → EN (%.60s)", source_lang, translated)
            return translated
        logger.warning("翻译结果过短, 使用原文: %.60s", text)
        return text
    except Exception as e:
        logger.warning("查询翻译失败 (%s→EN): %s, 使用原文", source_lang, e)
        return text


def translate_from_english(
    text: str,
    target_lang: str,
    llm: LLMClient,
) -> str:
    """将英文回答回译为用户的源语言.

    Args:
        text: 英文回答.
        target_lang: 目标语言代码 ('zh' | 'ko').
        llm: LLM 客户端.

    Returns:
        翻译后的文本. 失败或英文源语言时返回原文.
    """
    if not needs_translation(target_lang):
        return text

    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    system = _TRANSLATE_FROM_EN_SYSTEM.format(target_lang_name=lang_name)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Translate to {lang_name}:\n\n{text}"},
    ]

    try:
        result = llm.chat(messages, temperature=0.1, max_tokens=8192)
        translated = result.strip()
        if translated and len(translated) > 10:
            logger.info("回答回译: EN → %s (%d chars)", target_lang, len(translated))
            return translated
        logger.warning("回译结果过短, 使用英文原文")
        return text
    except Exception as e:
        logger.warning("回答回译失败 (EN→%s): %s, 使用英文原文", target_lang, e)
        return text
