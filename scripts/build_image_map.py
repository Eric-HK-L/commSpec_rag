#!/usr/bin/env python3
"""构建图片展示映射 — 方案 B (零重嵌入).

扫描 marked 数据集 raw.md, 提取图片, 生成 {spec_number: {figure_id: 相对路径}} 映射。

figure_id 用"短且稳定"的标识, 而非完整 alt: 3GPP 图片 alt 是几百字符的长描述,
splitter 切分时 [图: alt] 会被截断 (alt 后半段 + ] 丢失), 完整 alt 精确匹配失效。
故用 alt 前缀的 Figure 编号 (如 "Figure B.4-1", 编号在 alt 前部、截断时通常还在),
无编号时用 alt 前 80 字符兜底。跨 spec 用 spec_number 消歧。

存 data/processed/image_map.json。方案 A (重摄入保留链接) 后续可做。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_DISPLAYABLE = (".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp")


def figure_id(alt: str) -> str:
    """从 alt 提取短而稳定的标识 — 优先 Figure 编号, 否则前 80 字符."""
    alt = alt.strip()
    m = re.match(r"^(Figure\s+\S+)", alt)
    if m:
        return m.group(1)
    return alt[:80]


def spec_from_path(md: Path) -> str:
    r"""从路径 /(\d{2})_series/(\d{5})/ 提取 spec (如 38_series/38300 → 38.300)."""
    m = re.search(r"/(\d{2})_series/(\d{5})(?:-\d+)?/", str(md))
    if m:
        return f"{m.group(1)}.{m.group(2)[2:]}"
    return ""


def build_map(marked_dir: Path) -> dict[str, dict[str, str]]:
    img_map: dict[str, dict[str, str]] = {}
    skipped = 0
    missing = 0
    for md in marked_dir.rglob("raw.md"):
        spec = spec_from_path(md)
        text = md.read_text(encoding="utf-8", errors="replace")
        for alt, path in _IMG_RE.findall(text):
            alt = alt.strip()
            if not alt or not path:
                continue
            if not path.lower().endswith(_DISPLAYABLE):
                skipped += 1
                continue
            full = (md.parent / path).resolve()
            try:
                rel = full.relative_to(marked_dir.resolve())
            except ValueError:
                continue
            if not full.exists():
                missing += 1
                continue
            fid = figure_id(alt)
            img_map.setdefault(spec, {}).setdefault(fid, str(rel))
    total = sum(len(v) for v in img_map.values())
    print(f"  可展示图片映射: {total} 条 (spec {len(img_map)} 个) | 跳过非展示: {skipped} | 缺失: {missing}")
    return img_map


def main() -> None:
    marked = settings.documents_marked_dir
    print(f"扫描 marked 目录: {marked}")
    img_map = build_map(marked)
    out = settings.data_abs_dir / "processed" / "image_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(img_map, ensure_ascii=False), encoding="utf-8")
    print(f"图片映射已保存: {out}")


if __name__ == "__main__":
    main()
