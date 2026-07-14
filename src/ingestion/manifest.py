"""摄入清单模块 — 跟踪文档摄入状态，支持增量更新与版本管理.

核心设计:
  - Key = "{spec_number}|{release}" — 同 key 内版本升级覆盖，不同 key 共存
  - 版本比较: 3GPP 命名 i02 < i10 < i50 (解析数字部分)
  - 存储: JSON 文件，人类可读便于调试
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)

# 3GPP 文件名版本提取: "36322-i10" → "i10", "38101-f60" → "f60", "36101-id0" → "id0"
_3GPP_VERSION_RE = re.compile(r"[.-]([a-z]{1,3}\d{1,3}[a-z]?)$", re.IGNORECASE)


# ── 数据模型 ──

@dataclass
class SpecRecord:
    """单条规范摄入记录."""
    spec_number: str        # "36.322"
    release: str            # "R18"
    latest_version: str     # "i10"
    file_path: str          # 源 DOCX 路径
    sha256: str             # 文件内容哈希
    chunk_count: int        # 生成 chunk 数
    ingested_at: str        # ISO 时间戳


# ── 版本工具 ──

def parse_3gpp_version(filename_stem: str) -> str | None:
    """从文件名 stem 提取版本标识.

    "36322-i10" → "i10"
    "38101-f60" → "f60"
    "cover" → None
    """
    m = _3GPP_VERSION_RE.search(filename_stem)
    return m.group(1).lower() if m else None


def compare_versions(v1: str | None, v2: str | None) -> int:
    """比较 3GPP 版本号. 返回 -1/0/1 (v1 < / == / > v2).

    None 视为最低版本 (任何具体版本 > None).
    """
    if v1 == v2:
        return 0
    if v1 is None:
        return -1
    if v2 is None:
        return 1

    def _numeric(s: str) -> int:
        # 提取字母后的数字部分: "i10" → 10, "f60" → 60
        digits = re.sub(r"[a-z]", "", s, flags=re.IGNORECASE)
        return int(digits) if digits else 0

    n1, n2 = _numeric(v1), _numeric(v2)
    if n1 != n2:
        return -1 if n1 < n2 else 1
    # 数字相同时按字母排序 (如 i10a vs i10b)
    return -1 if v1 < v2 else (1 if v1 > v2 else 0)


# ── 清单管理 ──

class IngestionManifest:
    """摄入状态清单，持久化到 JSON 文件.

    线程不安全，调用方负责串行访问.
    """

    def __init__(self, manifest_path: str | Path | None = None):
        self._path = Path(manifest_path) if manifest_path else settings.manifest_path
        self._records: dict[str, SpecRecord] = {}  # key="{spec_number}|{release}"

    # ── 键构建 ──

    @staticmethod
    def make_key(spec_number: str, release: str) -> str:
        return f"{spec_number}|{release}"

    # ── 持久化 ──

    def load(self) -> None:
        """从 JSON 加载清单."""
        if not self._path.exists():
            logger.info("清单文件不存在，将以空清单启动: %s", self._path)
            self._records = {}
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            raw = data.get("specs", {})
            self._records = {}
            for key, item in raw.items():
                self._records[key] = SpecRecord(
                    spec_number=item["spec_number"],
                    release=item["release"],
                    latest_version=item["latest_version"],
                    file_path=item["file_path"],
                    sha256=item["sha256"],
                    chunk_count=item["chunk_count"],
                    ingested_at=item["ingested_at"],
                )
            logger.info("清单已加载: %d 条记录", len(self._records))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("清单文件损坏，重置空清单: %s", e)
            self._records = {}

    def save(self) -> None:
        """写入 JSON 文件."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, dict[str, Any]] = {
            "specs": {}
        }
        for key, rec in self._records.items():
            data["specs"][key] = asdict(rec)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("清单已保存: %d 条记录 → %s", len(self._records), self._path)

    # ── 查询 ──

    def find(self, spec_number: str, release: str) -> SpecRecord | None:
        """查找已摄入的规范记录."""
        return self._records.get(self.make_key(spec_number, release))

    def should_replace(self, spec_number: str, release: str, new_version: str | None) -> bool:
        """判断新版本是否应替换已摄入版本.

        条件: 已存在同一 spec+release，且新版本号 > 旧版本号.
        """
        existing = self.find(spec_number, release)
        if existing is None:
            return False  # 不存在 → 不是替换而是新增
        return compare_versions(new_version, existing.latest_version) > 0

    def has_same_hash(self, spec_number: str, release: str, sha256: str) -> bool:
        """检查同 key 记录的 SHA256 是否一致."""
        existing = self.find(spec_number, release)
        return existing is not None and existing.sha256 == sha256

    # ── 修改 ──

    def mark(
        self,
        spec_number: str,
        release: str,
        version: str | None,
        file_path: str,
        sha256: str,
        chunk_count: int,
    ) -> None:
        """记录/更新摄入状态."""
        key = self.make_key(spec_number, release)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._records[key] = SpecRecord(
            spec_number=spec_number,
            release=release,
            latest_version=version or "unknown",
            file_path=file_path,
            sha256=sha256,
            chunk_count=chunk_count,
            ingested_at=now,
        )

    def remove(self, spec_number: str, release: str) -> None:
        """移除记录."""
        self._records.pop(self.make_key(spec_number, release), None)

    def clear(self) -> None:
        """清空所有记录 (全量重建前)."""
        self._records.clear()

    # ── 孤儿检测 ──

    def get_orphaned_keys(
        self,
        existing_spec_releases: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """找出清单中有记录但当前文件系统中已不存在的 (spec_number, release).

        existing_spec_releases: 从当前 DOCX 文件扫描到的 {(spec_number, release), ...}
        """
        orphans: list[tuple[str, str]] = []
        for key in self._records:
            rec = self._records[key]
            if (rec.spec_number, rec.release) not in existing_spec_releases:
                orphans.append((rec.spec_number, rec.release))
        return orphans

    # ── 统计 ──

    @property
    def record_count(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"IngestionManifest({self.record_count} records)"
