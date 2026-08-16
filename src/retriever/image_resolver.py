"""图片展示 resolver — 方案 B (零重嵌入).

把 chunk 文本里的 `[图: <alt>]` 替换成可访问的图片 markdown `![<alt>](url)`,
让前端 ReactMarkdown 直接渲染图片。映射来自 data/processed/image_map.json
(由 scripts/build_image_map.py 离线生成)。

方案 A (重摄入保留链接) 后续可做, 本方案不改 chunk/不重嵌入。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

_IMG_TAG_RE = re.compile(r"\[图:\s*([^\]]+)\]")
_IMAGE_URL_PREFIX = "/images"

_instance: "ImageResolver | None" = None


class ImageResolver:
    """加载 alt→图片相对路径 映射, 提供文本替换."""

    def __init__(
        self,
        map_path: Path | None = None,
        image_url_prefix: str = _IMAGE_URL_PREFIX,
    ):
        self._map: dict[str, str] = {}
        self._prefix = image_url_prefix
        self._loaded = False
        self._path = map_path or (settings.data_abs_dir / "processed" / "image_map.json")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        if self._loaded:
            return True
        if not self._path.exists():
            logger.debug("图片映射不存在: %s", self._path)
            return False
        try:
            self._map = json.loads(self._path.read_text(encoding="utf-8"))
            self._loaded = True
            logger.info("图片展示映射已加载: %d 条", len(self._map))
        except Exception as e:
            logger.warning("图片映射加载失败: %s", e)
        return self._loaded

    def resolve(self, text: str) -> str:
        """把 text 中的 `[图: <alt>]` 替换成 `![<alt>](<prefix>/<rel>)`.

        无映射的 alt 原样保留 (不影响无图 chunk)。返回新字符串。
        """
        if not text or not self._loaded:
            return text

        def _repl(m: re.Match) -> str:
            alt = m.group(1).strip()
            rel = self._map.get(alt)
            if rel is None:
                return m.group(0)
            return f"![{alt}]({self._prefix}/{rel})"

        return _IMG_TAG_RE.sub(_repl, text)


def get_image_resolver() -> ImageResolver:
    """全局懒加载 resolver."""
    global _instance
    if _instance is None:
        _instance = ImageResolver()
        _instance.load()
    return _instance
