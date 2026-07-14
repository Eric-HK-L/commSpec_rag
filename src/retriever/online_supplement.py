"""在线搜索补充 — 离线检索不足时自动补充 Google/TSpec-LLM 在线结果.

触发条件:
  - 离线检索 Top-1 分数 < SIMILARITY_THRESHOLD
  - 离线结果数量 < MIN_RESULTS

来源:
  1. Google Custom Search (site:3gpp.org) — 需 GOOGLE_API_KEY + GOOGLE_CSE_ID
  2. TSpec-LLM API — 3GPP 官方 RAG 系统 (https://tspec-llm.3gpp.org)

结果标注 source='online'，不参与严格溯源校验.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class OnlineResult:
    """在线搜索结果."""
    title: str
    snippet: str
    url: str
    source: str  # "google" | "tspec-llm"
    score: float = 1.0


# ── Google Custom Search ──

class GoogleSearchProvider:
    """Google Custom Search — 限定 site:3gpp.org.

    需要环境变量:
      GOOGLE_API_KEY — Google Cloud API Key
      GOOGLE_CSE_ID  — Custom Search Engine ID
    """

    BASE_URL: ClassVar[str] = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str = "", cse_id: str = "", timeout: float = 5.0):
        self._api_key = api_key
        self._cse_id = cse_id
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key and self._cse_id)

    def search(self, query: str, num: int = 5) -> list[OnlineResult]:
        """执行 Google Custom Search."""
        if not self.enabled:
            logger.debug("Google Search 未配置 (GOOGLE_API_KEY/GOOGLE_CSE_ID)")
            return []

        import json as _json
        import urllib.request

        params = {
            "key": self._api_key,
            "cx": self._cse_id,
            "q": f"{query} site:3gpp.org",
            "num": min(num, 10),
        }
        url = self.BASE_URL + "?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items())

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read().decode())

            results: list[OnlineResult] = []
            for item in data.get("items", []):
                results.append(OnlineResult(
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    url=item.get("link", ""),
                    source="google",
                ))
            logger.info("Google Search: %d 条结果 (query=%s)", len(results), query[:60])
            return results
        except Exception as e:
            logger.warning("Google Search 失败: %s", e)
            return []


# ── TSpec-LLM API ──

class TSpecLLMProvider:
    """3GPP 官方 TSpec-LLM RAG API.

    端点: https://tspec-llm.3gpp.org/query (需确认)
    返回: 规范编号 + 段落引用 + 解释
    """

    DEFAULT_URL: ClassVar[str] = "https://tspec-llm.3gpp.org/query"

    def __init__(self, base_url: str = "", timeout: float = 10.0):
        self._base_url = base_url  # 空字符串表示未配置
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    def search(self, query: str, top_k: int = 5) -> list[OnlineResult]:
        """调用 TSpec-LLM API."""
        if not self.enabled:
            return []

        import json as _json
        import urllib.request

        payload = _json.dumps({
            "query": query,
            "max_results": top_k,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                self._base_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read().decode())

            results: list[OnlineResult] = []
            for item in data.get("results", []):
                results.append(OnlineResult(
                    title=item.get("spec", ""),
                    snippet=item.get("content", item.get("summary", "")),
                    url=item.get("url", ""),
                    source="tspec-llm",
                ))
            logger.info("TSpec-LLM: %d 条结果 (query=%s)", len(results), query[:60])
            return results
        except Exception as e:
            logger.debug("TSpec-LLM 不可用: %s", e)
            return []


# ── 在线补充编排器 ──

class OnlineSupplement:
    """在线搜索补充 — 离线不足时自动触发.

    用法:
        supplement = OnlineSupplement()
        online_results = supplement.supplement_if_needed(
            query="PDU Session Establishment",
            offline_best_score=0.55,
            offline_count=3,
        )
    """

    def __init__(
        self,
        google_api_key: str = "",
        google_cse_id: str = "",
        tspec_url: str = "",
        score_threshold: float = 0.6,
        count_threshold: int = 5,
        max_total: int = 5,
    ):
        self._google = GoogleSearchProvider(google_api_key, google_cse_id)
        self._tspec = TSpecLLMProvider(tspec_url)
        self._score_threshold = score_threshold
        self._count_threshold = count_threshold
        self._max_total = max_total

    @property
    def enabled(self) -> bool:
        return self._google.enabled or self._tspec.enabled

    def should_supplement(
        self, offline_best_score: float, offline_count: int
    ) -> bool:
        """判断是否需要在线补充."""
        if not self.enabled:
            return False
        if offline_count == 0:
            return True
        if offline_best_score < self._score_threshold:
            return True
        if offline_count < self._count_threshold:
            return True
        return False

    def supplement_if_needed(
        self,
        query: str,
        offline_best_score: float,
        offline_count: int,
        max_results: int | None = None,
    ) -> list[OnlineResult]:
        """条件触发在线搜索补充.

        Args:
            query: 原始查询
            offline_best_score: 离线检索最高分
            offline_count: 离线结果数量
            max_results: 在线最大结果数 (默认 self._max_total)

        Returns:
            在线补充结果列表
        """
        if not self.should_supplement(offline_best_score, offline_count):
            return []

        max_r = max_results if max_results is not None else self._max_total
        all_results: list[OnlineResult] = []

        # TSpec-LLM 优先 (官方来源)
        if self._tspec.enabled:
            t0 = time.time()
            results = self._tspec.search(query, max_r)
            logger.debug("TSpec-LLM 耗时: %.1fs", time.time() - t0)
            all_results.extend(results)

        # Google 补充
        remaining = max_r - len(all_results)
        if remaining > 0 and self._google.enabled:
            t0 = time.time()
            results = self._google.search(query, remaining)
            logger.debug("Google Search 耗时: %.1fs", time.time() - t0)
            all_results.extend(results)

        if all_results:
            logger.info(
                "在线补充: %d 条 (离线 best=%.2f, count=%d → 触发)",
                len(all_results), offline_best_score, offline_count,
            )
        return all_results[:max_r]

    def format_as_context(self, results: list[OnlineResult]) -> str:
        """将在线结果格式化为 LLM context 文本."""
        if not results:
            return ""

        lines = ["## 在线补充参考 (外部来源, 非本地规范库)\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"### 参考 {i}: {r.title}")
            lines.append(f"来源: {r.url}")
            lines.append(r.snippet)
            lines.append("")
        return "\n".join(lines)
