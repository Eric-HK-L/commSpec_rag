#!/usr/bin/env python3
"""3GPP 规范下载器 — 从 3GPP FTP 批量下载 DOCX 规范文档.

参考: SpecPilot 3gpp_extraction.py 的 FTP 遍历逻辑.
支持: --release / --series / --spec 筛选, 断点续传, dry-run 预览.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urljoin

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_specs")

# 3GPP FTP 镜像 (HTTP 访问)
BASE_URL = "https://www.3gpp.org/ftp/Specs/"

# Release → 路径前缀映射 (R18+ 在 latest/ 下按 Release 分目录)
RELEASE_PATHS = {
    "R15": "archive/",
    "R16": "archive/",
    "R17": "archive/",
    "R18": "latest/Rel-18/",
    "R19": "latest/Rel-19/",
}


class SpecDownloader:
    """3GPP 规范批量下载器."""

    def __init__(self, output_dir: str | None = None, timeout: int = 60):
        self._output = Path(output_dir) if output_dir else settings.documents_original_dir
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "3GPP-RAG-Downloader/1.0",
        })

    # ── 公共 API ──

    def download(
        self,
        release: str,
        series: int | None = None,
        spec: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """批量下载 3GPP 规范.

        Args:
            release: 例 "R18".
            series: 例 38 → 下载 38_series 全部.
            spec: 例 "38300" → 单篇下载.
            dry_run: 仅列出, 不下载.

        Returns:
            下载/预览的文件路径列表.
        """
        if spec:
            return self._download_spec(release, spec, dry_run)
        if series:
            return self._download_series(release, series, dry_run)
        logger.error("必须指定 --series 或 --spec")
        return []

    # ── 系列下载 ──

    def _download_series(self, release: str, series: int, dry_run: bool) -> list[str]:
        """下载整个 Series (如 38_series) 的所有规范."""
        prefix = RELEASE_PATHS.get(release, "archive/")
        series_dir = f"{series}_series"
        url = urljoin(BASE_URL, f"{prefix}{series_dir}/")

        logger.info("浏览: %s", url)
        links = self._list_links(url)
        docx_links = [link for link in links if link.lower().endswith(".docx") or link.lower().endswith(".zip")]

        if not docx_links:
            logger.warning("未找到 DOCX/ZIP 文件")
            return []

        downloaded: list[str] = []
        for link in docx_links:
            file_url = urljoin(url, link)
            local_path = self._local_path(release, series_dir, link.split("/")[-1])

            if dry_run:
                logger.info("[DRY-RUN] %s", file_url)
                downloaded.append(str(local_path))
                continue

            if self._download_file(file_url, local_path):
                downloaded.append(str(local_path))

        logger.info("Series %d: %d/%d 下载完成", series, len(downloaded), len(docx_links))
        return downloaded

    def _download_spec(self, release: str, spec: str, dry_run: bool) -> list[str]:
        """下载单篇规范 (如 TS 38.300). 3GPP FTP 按 Series 平铺存放."""
        prefix = RELEASE_PATHS.get(release, "archive/")
        series_num = spec[:2]
        series_dir = f"{series_num}_series"
        spec_dir = spec  # 如 "38300"

        # 尝试多种 URL 模式
        urls_to_try = [
            # 1. Series 目录平铺 (R18 标准路径)
            (urljoin(BASE_URL, f"{prefix}{series_dir}/"), spec),
            # 2. 单篇子目录 (旧版 FTP)
            (urljoin(BASE_URL, f"{prefix}{series_dir}/{spec_dir}/"), None),
            (urljoin(BASE_URL, f"{prefix}{spec_dir}/"), None),
        ]

        for base_url, filter_prefix in urls_to_try:
            try:
                links = self._list_links(base_url)
            except Exception:
                continue

            # 筛选文件: 若有 filter_prefix → 按文件名前缀匹配
            candidates = [link for link in links
                         if link.lower().endswith(".docx") or link.lower().endswith(".zip")]
            if filter_prefix:
                candidates = [link for link in candidates
                             if link.split("/")[-1].startswith(filter_prefix)]

            if candidates:
                # 选择最新版本 (文件名中数字最大的)
                candidates.sort(key=lambda x: self._version_key(x), reverse=True)
                chosen = candidates[0]
                # 提取纯文件名 (链接可能是完整 URL)
                filename = chosen.split("/")[-1]
                file_url = urljoin(base_url, chosen)
                local_path = self._local_path(release, series_dir, filename)

                if dry_run:
                    logger.info("[DRY-RUN] %s → %s", file_url, local_path)
                    return [str(local_path)]

                if self._download_file(file_url, local_path):
                    return [str(local_path)]

        logger.error("未找到规范 %s (release=%s)", spec, release)
        return []

    # ── 下载逻辑 ──

    def _download_file(self, url: str, local_path: Path) -> bool:
        """下载单个文件, 支持断点续传 (跳过已存在). ZIP 文件自动解压."""
        if local_path.exists():
            logger.debug("跳过: %s (已存在)", local_path.name)
            return True

        # 如果是 ZIP，检查是否已解压出 DOCX
        if local_path.suffix.lower() == ".zip":
            docx_dir = local_path.with_suffix("")
            if docx_dir.exists() and list(docx_dir.glob("*.docx")):
                logger.debug("跳过: %s (已解压)", local_path.name)
                return True

        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("下载: %s → %s", url, local_path.name)

        try:
            resp = self._session.get(url, timeout=self._timeout, stream=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        sys.stdout.write(f"\r  {pct}% ({downloaded}/{total})")
                        sys.stdout.flush()
            print()  # 换行

            # 自动解压 ZIP → DOCX
            if local_path.suffix.lower() == ".zip":
                self._extract_zip(local_path)

            return True
        except requests.RequestException as e:
            logger.error("下载失败 %s: %s", url, e)
            if local_path.exists():
                local_path.unlink()
            return False

    @staticmethod
    def _extract_zip(zip_path: Path) -> None:
        """解压 ZIP 到同名目录，仅提取 .docx 文件."""
        import zipfile
        extract_dir = zip_path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".docx"):
                    # 取文件名 (去掉路径前缀)
                    basename = Path(name).name
                    target = extract_dir / basename
                    if not target.exists():
                        with zf.open(name) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        logger.info("  解压: %s", basename)

    # ── 工具方法 ──

    def _list_links(self, url: str) -> list[str]:
        """列出 FTP 目录下的文件链接."""
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links: list[str] = []
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if href and href not in ("../", "./", "/"):
                links.append(href.rstrip("/"))
        return links

    def _local_path(self, release: str, series_dir: str, filename: str) -> Path:
        """生成本地存储路径."""
        return self._output / release / series_dir / filename

    @staticmethod
    def _version_key(filename: str) -> int:
        """从文件名提取版本排序键."""
        digits = "".join(ch for ch in filename if ch.isdigit())
        return int(digits) if digits else 0


# ══════════════════════ CLI ══════════════════════

def main():
    parser = argparse.ArgumentParser(description="3GPP 规范下载器")
    parser.add_argument("--release", default="R18", help="3GPP Release (默认: R18)")
    parser.add_argument("--series", type=int, help="Series 编号 (如 38)")
    parser.add_argument("--spec", help="单篇规范号 (如 38300)")
    parser.add_argument("--output", default=str(settings.documents_original_dir), help="输出目录 (默认: original/)")
    parser.add_argument("--dry-run", action="store_true", help="仅预览, 不下载")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP 超时秒数")
    args = parser.parse_args()

    downloader = SpecDownloader(output_dir=args.output, timeout=args.timeout)
    files = downloader.download(
        release=args.release,
        series=args.series,
        spec=args.spec,
        dry_run=args.dry_run,
    )
    if files:
        print(f"\n{'预览' if args.dry_run else '下载'}完成: {len(files)} 个文件")
    else:
        print("\n无文件")
        sys.exit(1)


if __name__ == "__main__":
    main()
