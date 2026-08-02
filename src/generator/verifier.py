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
                    f"答案引用了 TS {ref}，但该规范不在检索结果中，可能为幻觉"
                )
    
        # 引用编号有效性 — [i] 超出检索结果范围且未见于检索文本 (规范内部引用会出现在原文中)
        source_joined = "\n".join(s.text for s in sources if s.text)
        refs = [int(r) for r in re.findall(r"\[(\d{1,3})\]", answer)]
        invalid_refs = sorted({
            r for r in refs
            if r > len(sources) and f"[{r}]" not in source_joined
        })
        if invalid_refs:
            warnings.append(
                f"答案引用编号 {invalid_refs[:5]} 超出检索结果范围且未见于检索文本，可能为幻觉"
            )
    
        # 文本重合度检查 — 语言感知双信号 (性能优化后回答直接中文, 英文短语无法字面重合):
        #   1. 术语匹配 (主信号, 语言无关): 回答保留的英文专有术语须能在检索文本中找到
        #   2. 英文短语匹配 (补充信号): 英文内容较多时, 检索关键短语在回答中的字面覆盖率
        #   (实测中文回答含大量英文表格/References 时中文占比仅 ~14-26%, 阈值分流不可靠,
        #    故两信号并行, 任一通过即视为有据可依)
        zh_ratio = sum(1 for ch in answer if "\u4e00" <= ch <= "\u9fff") / max(len(answer), 1)
        zh_coverage, zh_warnings = self._verify_zh_overlap(answer, sources)
        coverage = zh_coverage
        if zh_ratio < 0.6:
            en_coverage = self._verify_en_overlap(answer, sources)
            if zh_coverage < 0.3:
                if en_coverage >= 0.3:
                    # 英文短语信号通过 → 撤销术语匹配警告
                    zh_warnings = [w for w in zh_warnings if "重合度" not in w]
                coverage = max(zh_coverage, en_coverage)
        warnings.extend(zh_warnings)
    
        return {
            "answer": answer,
            "verified": len(warnings) == 0,
            "warnings": warnings,
            "coverage": coverage,
        }
    
    @staticmethod
    def _verify_en_overlap(answer: str, sources: list[RetrievalResult]) -> float:
        """英文回答重合度: 检索文本关键短语在回答中的字面覆盖率."""
        answer_lower = answer.lower()
        cited_count = 0
        # 按英文句尾分割 (句号/问号/感叹号后跟空格), 避免 TS 38.300 / V18.4.0 被错误切分
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        total_sentences = len(sentences)
    
        for source in sources:
            if len(source.text) > 50:
                key_phrases = AnswerVerifier._extract_key_phrases(source.text, min_length=20)
                for phrase in key_phrases:
                    if phrase.lower() in answer_lower:
                        cited_count += 1
                        break
    
        if total_sentences > 0:
            coverage = min(1.0, cited_count / max(total_sentences, 1))
        else:
            coverage = 1.0
        return coverage
    
    @staticmethod
    def _verify_zh_overlap(answer: str, sources: list[RetrievalResult]) -> tuple[float, list[str]]:
        """中文回答溯源检查: 英文技术术语匹配率 (中文译文无法与英文原文字面重合).
    
        回答按 prompt 要求保留英文术语原文 (如 CIBF, DMRS, Section Type 5),
        这些专有术语应能在检索文本中找到 — 术语匹配率即溯源覆盖率.
        """
        source_text = "\n".join(s.text for s in sources if s.text).lower()
        # 专有术语: 全大写缩写 (UE/CIBF/O-RU/TS...) + "Section Type N" 模式
        terms = set(re.findall(r"\b[A-Z]{1,4}(?:-[A-Z0-9]{1,5})?\b", answer))
        terms |= set(re.findall(r"Section Type \d+", answer, re.IGNORECASE))
        # 过滤单字母与混入数字的伪术语 (如 "R18" 不会匹配, 纯大写缩写至少 2 字符)
        terms = {
            t for t in terms
            if (len(t) >= 2 and re.fullmatch(r"[A-Z]{2,4}(?:-[A-Z0-9]{1,5})?", t))
            or " " in t
        }
        if not terms:
            # 无英文术语可匹配, 仅依赖引用编号/TS 编号检查;
            # 无检索源时返回 0 (与英文短语路径一致, 避免空源误报高覆盖)
            return (0.0 if not sources else 1.0), []
        matched = sum(1 for t in terms if t.lower() in source_text)
        ratio = matched / len(terms)
        warnings: list[str] = []
        if ratio < 0.3:
            warnings.append(
                f"答案中的英文术语与检索结果重合度较低 ({ratio:.0%})，建议人工审核"
            )
        return ratio, warnings

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
