"""增量 Release 监控 — 检测 3GPP FTP 新文件并自动触发增量索引.

用法:
  - 定时检查:  startup/APScheduler 定时调度
  - 手动检查:  python -m src.ingestion.release_monitor --check

工作流:
  1. 扫描 data/documents/ 下的 DOCX 文件 (SHA256 hash)
  2. 对比 manifest ingestion_state.json 中的已处理记录
  3. 发现新文件/变更文件 → 标记为待摄入
  4. 可选自动触发: 调用 bulk_ingest.py --incremental
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocFile:
    """文档文件描述."""
    path: Path
    sha256: str
    size: int
    mtime: float
    series: int = 0
    spec_number: str = ""
    release: str = ""


@dataclass
class ChangeReport:
    """变更报告."""
    new_files: list[DocFile] = field(default_factory=list)
    modified_files: list[DocFile] = field(default_factory=list)
    deleted_keys: list[str] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_keys)

    @property
    def total_changes(self) -> int:
        return len(self.new_files) + len(self.modified_files) + len(self.deleted_keys)


class ReleaseMonitor:
    """3GPP 文档变更监控器.

    对比文档目录与 manifest，检测新增/修改/删除。
    """

    def __init__(
        self,
        doc_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        on_change: Callable[[ChangeReport], None] | None = None,
    ):
        self._doc_dir = Path(doc_dir) if doc_dir else settings.documents_original_dir
        self._manifest_path = Path(manifest_path) if manifest_path else settings.manifest_path
        self._on_change = on_change
        self._last_manifest: dict = {}

    # ── 文件扫描 ──

    def _sha256_file(self, path: Path, chunk_size: int = 65536) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _parse_spec_info(self, filename: str) -> tuple[int, str, str]:
        """从文件名提取 series / spec_number / release.
        文件名格式: TS_38.300_R18_v17.0.0.docx 或 38300-rel18.docx
        """
        name = Path(filename).stem
        series = 0
        spec_num = ""
        release = ""

        # TS 38.300 / TR 23.501 格式
        m = __import__("re").search(r"(?:TS|TR)\s*(\d{2})\.(\d{3})", name, __import__("re").IGNORECASE)
        if m:
            series = int(m.group(1))
            spec_num = f"{m.group(1)}{m.group(2)}"
        else:
            # 38300-rel18 格式
            m = __import__("re").search(r"(\d{4,5})", name)
            if m:
                spec_num = m.group(1)
                series = int(spec_num[:2])

        # Release
        m = __import__("re").search(r"[Rr]el(?:ease)?[._-]?(\d{2})", name)
        if m:
            release = f"R{m.group(1)}"

        return series, spec_num, release

    def scan_documents(self) -> dict[str, DocFile]:
        """扫描文档目录, 返回 {文件路径字符串: DocFile}."""
        if not self._doc_dir.exists():
            logger.warning("文档目录不存在: %s", self._doc_dir)
            return {}

        files: dict[str, DocFile] = {}
        for docx_path in sorted(self._doc_dir.rglob("*.docx")):
            try:
                series, spec, rel = self._parse_spec_info(docx_path.name)
                stat = docx_path.stat()
                files[str(docx_path)] = DocFile(
                    path=docx_path,
                    sha256=self._sha256_file(docx_path),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    series=series,
                    spec_number=spec,
                    release=rel,
                )
            except Exception as e:
                logger.warning("扫描文件失败 %s: %s", docx_path, e)

        return files

    # ── Manifest ──

    def load_manifest(self) -> dict:
        """加载 manifest."""
        if not self._manifest_path.exists():
            return {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                self._last_manifest = json.load(f)
            return self._last_manifest
        except Exception as e:
            logger.warning("Manifest 读取失败: %s", e)
            return {}

    def save_manifest(self, manifest: dict) -> None:
        """保存 manifest."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── 变更检测 ──

    def detect_changes(self) -> ChangeReport:
        """检测文档变更.

        Returns:
            ChangeReport 包含 new_files / modified_files / deleted_keys
        """
        current_files = self.scan_documents()
        manifest = self.load_manifest()
        report = ChangeReport()

        # 已处理记录
        processed: dict[str, dict] = manifest.get("files", {})
        processed_hashes: dict[str, str] = {}  # manifest_key → sha256

        for key, info in processed.items():
            # key 格式: "data/documents/R18/38_series/TS_38.300_R18.docx"
            sha = info.get("sha256", "")
            abs_path = info.get("path", key)
            processed_hashes[abs_path] = sha

        # 构建当前文件 → manifest key 的映射
        current_by_key: dict[str, DocFile] = {}
        current_by_abs: dict[str, DocFile] = {}
        for path_str, doc in current_files.items():
            current_by_abs[path_str] = doc
            # 尝试匹配 manifest key
            for mkey in processed:
                if path_str.endswith(mkey) or mkey.endswith(path_str) or path_str == mkey:
                    current_by_key[mkey] = doc
                    break
            else:
                current_by_key[path_str] = doc

        # 查找新增/修改
        for mkey, doc in current_by_key.items():
            old_sha = processed_hashes.get(mkey, "")
            if not old_sha:
                report.new_files.append(doc)
            elif old_sha != doc.sha256:
                report.modified_files.append(doc)

        # 查找删除 (manifest 中有但磁盘上无)
        for mkey in processed:
            if mkey not in current_by_key:
                report.deleted_keys.append(mkey)

        logger.info(
            "变更检测: +%d ~%d -%d (共 %d 文件)",
            len(report.new_files), len(report.modified_files),
            len(report.deleted_keys), len(current_files),
        )

        self._last_manifest = manifest
        return report

    # ── 自动处理 ──

    def check_and_process(
        self,
        auto_index: bool = False,
        on_progress: Callable[[str], None] | None = None,
    ) -> ChangeReport:
        """检测变更并可选自动触发索引.

        Args:
            auto_index: 是否自动触发 bulk_ingest.py --incremental
            on_progress: 进度回调

        Returns:
            ChangeReport
        """
        report = self.detect_changes()

        if on_progress:
            on_progress(
                f"检测完成: +{len(report.new_files)} "
                f"~{len(report.modified_files)} "
                f"-{len(report.deleted_keys)} 文件"
            )

        if report.has_changes and self._on_change:
            try:
                self._on_change(report)
            except Exception as e:
                logger.error("变更回调失败: %s", e)

        # 自动索引
        if auto_index and (report.new_files or report.modified_files):
            self._trigger_ingestion(on_progress)

        return report

    def _trigger_ingestion(
        self, on_progress: Callable[[str], None] | None = None
    ) -> None:
        """触发增量摄入."""
        project_root = Path(__file__).resolve().parent.parent.parent
        script = project_root / "scripts" / "bulk_ingest.py"
        venv_python = sys.executable

        if not script.exists():
            logger.warning("摄入脚本不存在: %s", script)
            return

        cmd = [venv_python, str(script), "--incremental"]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(project_root),
                text=True,
            )
            if on_progress:
                on_progress(f"摄入已触发 (PID={proc.pid})")
            logger.info("增量摄入已触发: PID=%d", proc.pid)
        except Exception as e:
            logger.error("触发摄入失败: %s", e)


# ── CLI ──

def _cli_main() -> None:
    """命令行入口: python -m src.ingestion.release_monitor --check [--auto]"""
    import argparse

    parser = argparse.ArgumentParser(description="3GPP 文档变更监控")
    parser.add_argument("--check", action="store_true", help="检测变更")
    parser.add_argument("--auto", action="store_true", help="自动触发增量索引")
    parser.add_argument("--doc-dir", default=str(settings.documents_original_dir), help="文档目录")
    parser.add_argument("--manifest", default=str(settings.manifest_path))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    monitor = ReleaseMonitor(
        doc_dir=args.doc_dir,
        manifest_path=args.manifest,
    )

    if args.check:
        report = monitor.check_and_process(auto_index=args.auto)
        if report.has_changes:
            print(f"\n变更报告 ({time.strftime('%Y-%m-%d %H:%M:%S')}):")
            for f in report.new_files:
                print(f"  + {f.path.name} ({f.spec_number}, {f.release})")
            for f in report.modified_files:
                print(f"  ~ {f.path.name} ({f.spec_number}, {f.release})")
            for k in report.deleted_keys:
                print(f"  - {k}")
            print(f"\n共 {report.total_changes} 项变更")
        else:
            print("无变更")
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
