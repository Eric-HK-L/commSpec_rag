"""图片展示 resolver — 方案 B (零重嵌入).

把 chunk 文本里的 `[图: <alt>]` 替换成可访问的图片 markdown `![<alt>](url)`。
映射来自 data/processed/image_map.json ({spec: {figure_id: 相对路径}}, 由
scripts/build_image_map.py 生成)。

figure_id 用短标识 (Figure 编号 / alt 前 80 字符), 而非完整 alt — 因 3GPP 图片 alt
是长描述, splitter 切分时 [图: alt] 会被截断, 完整 alt 匹配失效。resolve 需传 spec_number
用于跨 spec 消歧。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

_IMG_TAG_RE = re.compile(r"\[图:\s*([^\]\n]+)\]?")
_IMAGE_URL_PREFIX = "/images"

_instance: "ImageResolver | None" = None


def _figure_id(alt: str) -> str:
    alt = alt.strip()
    m = re.match(r"^(Figure\s+\S+)", alt)
    if m:
        return m.group(1).rstrip(":：")
    return alt[:80]


class ImageResolver:
    """加载 {spec: {figure_id: rel}} 映射, 提供文本替换."""

    def __init__(
        self,
        map_path: Path | None = None,
        image_url_prefix: str = _IMAGE_URL_PREFIX,
    ):
        self._map: dict[str, dict[str, str]] = {}
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
            logger.info("图片展示映射已加载: %d 个 spec", len(self._map))
        except Exception as e:
            logger.warning("图片映射加载失败: %s", e)
        return self._loaded

    def resolve(self, text: str, spec_number: str = "") -> str:
        """把 text 中的 `[图: <alt>]` 替换成 `![<alt>](<prefix>/<rel>)`.

        匹配 figure_id (短标识), 传 spec_number 用于跨 spec 消歧; 无映射时原样保留。
        """
        if not text or not self._loaded:
            return text
        spec_map = self._map.get(spec_number or "", {})

        def _repl(m: re.Match) -> str:
            alt = m.group(1).strip()
            fid = _figure_id(alt)
            rel = spec_map.get(fid)
            if rel is None:
                return m.group(0)
            # 用短 figure_id 作 alt (而非完整长描述): 长 alt 会占满 source.text 的
            # [:500] 截断窗口, 使图片 URL 落在截断之后被丢掉。
            return f"![{fid}]({self._prefix}/{rel})"

        return _IMG_TAG_RE.sub(_repl, text)


def get_image_resolver() -> ImageResolver:
    """全局懒加载 resolver."""
    global _instance
    if _instance is None:
        _instance = ImageResolver()
        _instance.load()
    return _instance
