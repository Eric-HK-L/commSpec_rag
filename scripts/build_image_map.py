#!/usr/bin/env python3
"""构建图片展示映射 — {alt 描述 → 图片相对路径} (方案 B, 零重嵌入).

扫描 marked 数据集 raw.md, 提取 `![alt](图片路径)`, 存 data/processed/image_map.json。
用于后端把 chunk 里的 `[图: <alt>]` 替换成可访问的图片 URL, 实现"命中含图 chunk 时贴图"。

方案 B 不重嵌入: chunk 文本不动, 仅靠 alt 描述反查图片路径; 方案 A (重摄入保留链接) 后续可做。
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
# 浏览器可直接显示的格式 (排除 emf 等矢量/专用格式)
_DISPLAYABLE = (".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp")


def build_map(marked_dir: Path) -> dict[str, str]:
    img_map: dict[str, str] = {}
    skipped_emf = 0
    missing = 0
    for md in marked_dir.rglob("raw.md"):
        text = md.read_text(encoding="utf-8", errors="replace")
        for alt, path in _IMG_RE.findall(text):
            alt = alt.strip()
            if not alt or not path:
                continue
            if not path.lower().endswith(_DISPLAYABLE):
                skipped_emf += 1
                continue
            full = (md.parent / path).resolve()
            try:
                rel = full.relative_to(marked_dir.resolve())
            except ValueError:
                continue
            if not full.exists():
                missing += 1
                continue
            img_map.setdefault(alt, str(rel))
    print(f"  可展示图片映射: {len(img_map)} 条 | 跳过非展示格式: {skipped_emf} | 文件缺失: {missing}")
    return img_map


def main() -> None:
    marked = settings.documents_marked_dir
    print(f"扫描 marked 目录: {marked}")
    img_map = build_map(marked)
    out = settings.data_abs_dir / "processed" / "image_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(img_map, ensure_ascii=False), encoding="utf-8")
    print(f"图片映射已保存: {out} ({len(img_map)} 条)")


if __name__ == "__main__":
    main()
