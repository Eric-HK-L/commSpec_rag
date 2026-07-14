#!/usr/bin/env python3
"""全量/增量摄入脚本 — 遍历 DOCX，走 extract→split→embed→insert.

模式:
  python scripts/bulk_ingest.py                    # 默认：增量模式
  python scripts/bulk_ingest.py --full-rebuild     # 全量重建: drop collection + 清 manifest

版本管理:
  - 大版本共存: R18 和 R19 同一规范各有独立 key (spec_number|release)
  - 小版本覆盖: 同 release 内新版替换旧版 (基于文件内容 SHA256 + 版本号)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.embedder import BatchEmbedder  # noqa: E402
from src.ingestion.embedding_cache import EmbeddingCache  # noqa: E402
from src.ingestion.extractor import DoclingExtractor  # noqa: E402
from src.ingestion.manifest import (  # noqa: E402
    IngestionManifest,
    compare_versions,
    parse_3gpp_version,
)
from src.ingestion.mps_embedder import MPSChunkedEmbedder  # noqa: E402
from src.ingestion.splitter import HeaderAwareSplitter  # noqa: E402
from src.retriever.milvus_store import MilvusStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bulk_ingest")


# ── 工具函数 ──

def compute_sha256(filepath: Path) -> str:
    """计算文件 SHA256 哈希."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# 3GPP 规范编号解析: "36322" → "36.322", "38101" → "38.101"
# 从纯数字 stem 提取规范编号: "21900" → ("21", "900")
_SPEC_PARSE_RE = re.compile(r"^(\d{2})(\d{2,3})$")
# 从含子规范号/分卷标记的 stem 提取前 5 位数字: "23700-18-i00" → "23700"
_SPEC5_DIGIT_RE = re.compile(r"^(\d{5})")
# 分卷后缀: "_s00-07", "_sA.1-A.3", "_S0-7" 等
_SECTION_SUFFIX_RE = re.compile(r"_[sS].*$")


def parse_spec_from_stem(stem: str) -> tuple[str, str | None]:
    """从文件名 stem 解析规范编号和版本.

    支持格式:
      "36322-i10"           → ("36.322", "i10")    # 标准格式
      "23700-18-i00"         → ("23.700", "i00")    # 子规范号 (23.700-18)
      "38101-2-i40"          → ("38.101", "i40")    # 子部分
      "36101-id0_s00-07"     → ("36.101", "id0")    # 分卷文档
      "38533-i80_s00-s04"    → ("38.533", "i80")    # 分卷文档 (多段范围)
      "38133-ie0_sA.1-A.3"   → ("38.133", "ie0")    # Annex 分卷
      "cover"                → ("", None)
    """
    # 1. 先去分卷后缀 (必须在提取版本前完成，否则 _s05 被误认为版本)
    base = _SECTION_SUFFIX_RE.sub("", stem)

    # 2. 从剩余部分提取版本
    version = parse_3gpp_version(base)

    # 3. 去掉版本后缀得到纯规范编号部分
    if version:
        idx = base.rfind(f"-{version}")
        if idx >= 0:
            base = base[:idx]
        else:
            idx = base.rfind(f"_{version}")
            if idx >= 0:
                base = base[:idx]

    # 4a) 标准格式: 纯 5 位数字 "36322" → "36.322"
    m = _SPEC_PARSE_RE.match(base)
    if m:
        return f"{m.group(1)}.{m.group(2)}", version

    # 4b) 含子规范号/子部分: "23700-18" 或 "38101-2" → 取前 5 位
    m = _SPEC5_DIGIT_RE.match(base)
    if m:
        digits = m.group(1)
        return f"{digits[:2]}.{digits[2:]}", version

    return "", version


def find_docx_files(doc_dir: str) -> list[Path]:
    """收集统一文档目录下所有 DOCX 文件，按 Series 排序.

    目录结构: data/documents/R18/{21_series/,22_series/,...}
    跳过封面页: _cover.docx 仅含标题/版本信息，无规范正文。
    """
    root = Path(doc_dir)
    if not root.exists():
        logger.error("目录不存在: %s", root)
        return []
    files = sorted(root.rglob("*.docx"), key=lambda p: (p.parent.name, p.name))
    # 跳过仅封面页（无正文内容）
    files = [f for f in files if "_cover" not in f.stem.lower() and "cover" != f.stem.lower()]
    return files


# ── 单篇处理 ──

def process_single_docx(
    docx: Path,
    extractor: DoclingExtractor,
    splitter: HeaderAwareSplitter,
) -> tuple[list, str, str, str, str]:
    """处理单篇 DOCX: 提取 + 分块.

    返回 (chunks, spec_number, release, version, sha256).
    spec_number/version 空字符串表示解析失败.
    """
    sha = compute_sha256(docx)
    spec_number, version = parse_spec_from_stem(docx.stem)
    result = extractor.extract_file(docx)
    # 优先用 DOCX 内容头检测的 release，其次从目录名推断
    release = result.release
    if not release:
        # 尝试从父目录名推断 (如 data/documents/R18/38_series/xxx.docx → R18)
        parent_names = [p.name for p in docx.parents]
        for pn in parent_names:
            if pn.upper().startswith("R") and len(pn) <= 4 and pn[1:].isdigit():
                release = pn.upper()
                break
    if not release:
        logger.warning("无法检测 Release, 默认 R18: %s", docx.name)
        release = "R18"

    if not result.markdown:
        return [], spec_number, release, version, sha

    doc_meta = {
        "doc_id": docx.stem,
        "series": int(spec_number.split(".")[0]) if spec_number else 0,
        "spec_number": spec_number,
        "release": release,
    }
    chunks = splitter.split_document(result.markdown, doc_meta)
    return chunks, spec_number, release, version, sha


# ── 增量模式主逻辑 ──

def run_incremental(
    docx_files: list[Path],
    extractor: DoclingExtractor,
    splitter: HeaderAwareSplitter,
    store: MilvusStore,
    manifest: IngestionManifest,
) -> dict:
    """增量摄入: 只处理新增/修改的文件."""
    stats = {"total": len(docx_files), "skipped": 0, "replaced": 0, "new": 0, "failed": 0, "chunks": 0}
    all_chunks: list = []

    # 收集当前文件系统的 spec_number+release (从目录名推断)
    current_specs: set[tuple[str, str]] = set()
    for docx in docx_files:
        sn, _ = parse_spec_from_stem(docx.stem)
        # 尝试从父目录名推断 release
        release = "R18"
        for pn in [p.name for p in docx.parents]:
            if pn.upper().startswith("R") and len(pn) <= 4 and pn[1:].isdigit():
                release = pn.upper()
                break
        current_specs.add((sn, release))

    for i, docx in enumerate(docx_files):
        try:
            chunks, spec_number, release, version, sha = process_single_docx(
                docx, extractor, splitter,
            )
            if not spec_number:
                logger.warning("[%d/%d] 无法解析规范编号: %s", i + 1, len(docx_files), docx.name)
                stats["failed"] += 1
                continue

            manifest.make_key(spec_number, release)

            # 判断是否需要处理
            existing = manifest.find(spec_number, release)

            if existing is not None:
                # 已存在 → 检查是否需要替换
                if compare_versions(version, existing.latest_version) <= 0:
                    if sha == existing.sha256:
                        logger.debug("[%d/%d] 跳过（未变）: %s", i + 1, len(docx_files), docx.name)
                        stats["skipped"] += 1
                        continue
                    else:
                        logger.warning(
                            "[%d/%d] 版本未升级但内容已变 (%s): %s, 跳过",
                            i + 1, len(docx_files), version, docx.name,
                        )
                        stats["skipped"] += 1
                        continue

                # 新版本 > 旧版本 → 替换
                logger.info(
                    "[%d/%d] 版本升级 %s→%s: %s",
                    i + 1, len(docx_files), existing.latest_version, version, docx.name,
                )
                count = store.delete_by_filter(
                    f'spec_number == "{spec_number}" && release == "{release}"'
                )
                logger.info("  已删除旧 chunks: %d 条", count)
                stats["replaced"] += 1
            else:
                logger.info("[%d/%d] 新增: %s (%s %s)", i + 1, len(docx_files), docx.name, spec_number, release)
                stats["new"] += 1

            if not chunks:
                logger.warning("[%d/%d] 空内容: %s", i + 1, len(docx_files), docx.name)
                manifest.mark(spec_number, release, version, str(docx), sha, 0)
                stats["failed"] += 1
                continue

            all_chunks.extend(chunks)
            manifest.mark(spec_number, release, version, str(docx), sha, len(chunks))

            if (i + 1) % 50 == 0:
                logger.info(
                    "[%d/%d] 进度: %d 跳过 / %d 新增 / %d 替换 / %d chunks",
                    i + 1, len(docx_files),
                    stats["skipped"], stats["new"], stats["replaced"], len(all_chunks),
                )

        except Exception as e:
            logger.error("[%d/%d] 处理失败 %s: %s", i + 1, len(docx_files), docx.name, e)
            stats["failed"] += 1

    # 清理孤儿: 清单中有但文件系统已删除的
    orphaned = manifest.get_orphaned_keys(current_specs)
    for spec_number, release in orphaned:
        logger.info("清理孤儿: %s|%s (源文件已删除)", spec_number, release)
        store.delete_by_filter(
            f'spec_number == "{spec_number}" && release == "{release}"'
        )
        manifest.remove(spec_number, release)

    manifest.save()
    stats["chunks"] = len(all_chunks)
    return stats, all_chunks


# ── 全量重建 ──

def run_full_rebuild(
    docx_files: list[Path],
    extractor: DoclingExtractor,
    splitter: HeaderAwareSplitter,
    store: MilvusStore,
    manifest: IngestionManifest,
) -> dict:
    """全量重建: drop collection + 清 manifest，然后摄入全部."""
    stats = {"total": len(docx_files), "skipped": 0, "replaced": 0, "new": 0, "failed": 0, "chunks": 0}
    all_chunks: list = []

    manifest.clear()
    logger.info("已清空清单，开始全量重建 (drop collection)")

    for i, docx in enumerate(docx_files):
        try:
            chunks, spec_number, release, version, sha = process_single_docx(
                docx, extractor, splitter,
            )
            if not spec_number or not chunks:
                stats["failed"] += 1
                if spec_number:
                    manifest.mark(spec_number, release, version, str(docx), sha, len(chunks))
                continue

            all_chunks.extend(chunks)
            manifest.mark(spec_number, release, version, str(docx), sha, len(chunks))
            stats["new"] += 1

            if (i + 1) % 50 == 0:
                logger.info(
                    "[%d/%d] 已提取 %d chunks (%d 篇)",
                    i + 1, len(docx_files), len(all_chunks), stats["new"],
                )

        except Exception as e:
            logger.error("[%d/%d] 处理失败 %s: %s", i + 1, len(docx_files), docx.name, e)
            stats["failed"] += 1

    manifest.save()
    stats["chunks"] = len(all_chunks)
    return stats, all_chunks


def _should_use_mps() -> bool:
    """判断是否应使用 MPS 加速嵌入.

    条件:
      1. MPS 硬件可用 (Apple Silicon)
      2. 用户未显式禁用 (EMBEDDING_DEVICE 不是 cpu)
      3. 大规模嵌入 (>5000 texts) — 小批量无需 fork overhead
    """
    return (
        MPSChunkedEmbedder.is_mps_available()
        and settings.embedding_device.lower() != "cpu"
    )


def _embed_via_subprocess(texts: list[str]) -> "np.ndarray":
    """通过独立 subprocess 进行 MPS 嵌入, 隔离 gRPC 状态 + 避免 Metal 多进程死锁.

    双重隔离:
    1. subprocess 隔离: 父进程已 import pymilvus (gRPC 全局状态),
       spawn 子进程会 import __main__ 连带引入 gRPC → FD 污染。
       本函数启动独立 subprocess, __main__ 干净。
    2. workers=1: Apple MPS 不支持多进程 GPU — 多进程同时调用
       [MTLCommandBuffer waitUntilCompleted] 触发 Metal 调度器死锁。
       单 worker GPU 内部并行已足够快 (~75 t/s on M4 Max, 3GPP 长文本)。

    流程: 文本写入临时 pickle → subprocess 独立嵌入 → 读回 .npy。
    超时动态计算: 按保守 30 t/s + 600s 模型加载余量, 随数据规模自适应。
    """
    import numpy as np
    import threading

    script = PROJECT_ROOT / "scripts" / "_mps_embed_subprocess.py"

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        pickle.dump(texts, f)
        texts_file = f.name

    embeddings_file = texts_file.replace(".pkl", ".npy")

    try:
        cmd = [
            sys.executable, str(script),
            texts_file, embeddings_file,
            "--device", "mps",
        ]

        # 动态超时: 保守估计 1 t/s (36.331/38.331 长文本实测 1 t/s) + 1800s 模型加载
        timeout = max(600, int(len(texts) / 1) + 1800)
        logger.info(
            "启动 MPS 嵌入子进程 (%d texts, timeout=%ds ≈ %.0fmin)",
            len(texts), timeout, timeout / 60,
        )

        # ── Popen 实时流式输出 ──
        # 不用 subprocess.run(capture_output=True): 输出只在进程退出后可见,
        # 超时后丢失全部进度信息。Popen + 线程实时转发 stderr → 日志。
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stderr_lines: list[str] = []

        def _read_stderr():
            for line in process.stderr:
                stripped = line.rstrip()
                if stripped and "DeprecationWarning" not in stripped:
                    logger.info("[mps-sub] %s", stripped)
                stderr_lines.append(line)

        reader = threading.Thread(target=_read_stderr, daemon=True)
        reader.start()

        try:
            stdout, _ = process.communicate(timeout=timeout)
            reader.join(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join(timeout=5)
            logger.error(
                "MPS 子进程超时 (%ds, %d texts, 最后输出见上方日志)",
                timeout, len(texts),
            )
            raise

        # 打印 stdout
        for line in stdout.splitlines():
            if line.strip():
                logger.info("[mps-sub] %s", line)

        if process.returncode != 0:
            last_lines = "".join(stderr_lines[-10:])
            raise RuntimeError(f"MPS 子进程返回 {process.returncode}: {last_lines}")

        embeddings = np.load(embeddings_file)
        logger.info("嵌入结果已加载: shape=%s, dtype=%s", embeddings.shape, embeddings.dtype)
        return embeddings
    finally:
        for f in [texts_file, embeddings_file]:
            try:
                os.unlink(f)
            except OSError:
                pass


# ── 嵌入 + 入库 ──

def embed_and_insert(
    all_chunks: list,
    store: MilvusStore,
    skip_chunks: int = 0,
) -> int:
    """MPS 分段嵌入 + 逐段入库 Milvus。

    每段嵌入后立即入库 (断点续传: 段失败不影响已入库段)。
    skip_chunks: 跳过前 N 条 (从 Milvus 恢复时使用)。
    MPS 内存由 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 限制 (约 25GB),
    子进程内部每 500 文本重载模型回收 MPS 缓存。段间进程退出由 OS
    回收 wired 内存。实测 batch_size=4, wired ~7GB。
    """
    if not all_chunks:
        logger.warning("无 chunk 可入库")
        return 0

    store.connect()

    total = len(all_chunks)
    # BGE-M3 1024-dim 向量约 4KB/chunk, 5000 条 ≈ 35MB gRPC 消息, 安全低于 64MB 上限
    segment_size = 5000
    num_segments = (total + segment_size - 1) // segment_size
    skip_segments = skip_chunks // segment_size

    if skip_segments > 0:
        logger.info("断点续传: 跳过前 %d/%d 段 (%d chunks)", skip_segments, num_segments, skip_chunks)

    logger.info("开始 MPS 分段嵌入 %d chunks (%d 段 × ≤%d 条/段)", total, num_segments, segment_size)
    t_embed = time.time()

    use_mps = _should_use_mps()
    logger.info("嵌入后端: %s", "MPS (子进程)" if use_mps else f"CPU (device={settings.resolved_embedding_device})")

    inserted_total = 0

    for seg_idx in range(skip_segments, num_segments):
        start = seg_idx * segment_size
        end = min(start + segment_size, total)
        seg_texts = [c.text for c in all_chunks[start:end]]
        seg_count = len(seg_texts)

        logger.info("[段 %d/%d] 编码 %d 条文本...", seg_idx + 1, num_segments, seg_count)
        t_seg = time.time()

        if use_mps:
            seg_emb = _embed_via_subprocess(seg_texts)
        else:
            sqlite_cache = EmbeddingCache()
            embedder = BatchEmbedder(
                batch_size=32,
                sqlite_cache=sqlite_cache,
                on_progress=lambda done, total: logger.info(
                    "[CPU嵌入] %d/%d (%.0f%%)", done, total, 100 * done / total
                ) if done % 5000 == 0 or done == total else None,
            )
            seg_emb = embedder.embed_batch(seg_texts)

        seg_elapsed = time.time() - t_seg
        logger.info(
            "[段 %d/%d] 嵌入完成: %d vectors, %.1fs (%.0f t/s)",
            seg_idx + 1, num_segments, seg_count, seg_elapsed,
            seg_count / seg_elapsed if seg_elapsed > 0 else 0,
        )

        # 逐段入库 (断点续传: 段失败不影响已入库段)
        seg_chunks = all_chunks[start:end]
        for c, emb in zip(seg_chunks, seg_emb):
            c.embedding = emb
        n = store.insert(seg_chunks)
        inserted_total += n
        logger.info(
            "[段 %d/%d] 入库: %d chunks → Milvus (累计 %d/%d)",
            seg_idx + 1, num_segments, n, inserted_total, total,
        )

    logger.info("全部嵌入+入库完成: %d chunks, %.1fs", inserted_total, time.time() - t_embed)
    return inserted_total


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(description="3GPP 规范摄入脚本")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="全量重建: drop collection + 清空清单后重新摄入全部",
    )
    parser.add_argument(
        "--doc-dir",
        default=str(settings.documents_abs_dir),
        help="文档根目录 (默认: 来自 DOCUMENTS_DIR 配置)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="从 checkpoint 恢复 (跳过提取, 直接嵌入+入库; 自动检测 Milvus 断点续传)",
    )
    args = parser.parse_args()

    t_start = time.time()
    doc_dir = args.doc_dir
    full_rebuild = args.full_rebuild
    checkpoint_path = settings.checkpoint_path

    docx_files = find_docx_files(doc_dir)
    logger.info("找到 %d 个 DOCX 文件", len(docx_files))

    if not docx_files:
        logger.error("无 DOCX 文件")
        return

    extractor = DoclingExtractor()
    splitter = HeaderAwareSplitter(max_chunk_chars=2500, chunk_overlap_chars=100)
    manifest = IngestionManifest()
    manifest.load()

    store = MilvusStore()

    # ── 提取阶段 (或从 checkpoint 恢复) ──
    if args.resume_from_checkpoint and checkpoint_path.exists():
        logger.info("从 checkpoint 恢复 chunks: %s", checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
        all_chunks = saved["chunks"]
        stats = saved["stats"]
        logger.info("已恢复 %d chunks (DOCX: %d)", len(all_chunks), stats["total"])
    else:
        store.connect()
        if full_rebuild:
            store.create_collection(drop_existing=True)
            logger.info("已重建 collection")
            stats, all_chunks = run_full_rebuild(docx_files, extractor, splitter, store, manifest)
        else:
            store.create_collection(drop_existing=False)
            stats, all_chunks = run_incremental(docx_files, extractor, splitter, store, manifest)
        store.disconnect()

        # 保存 checkpoint (提取最耗时, 避免嵌入失败后重提取)
        if all_chunks:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"chunks": all_chunks, "stats": stats}, f)
            logger.info("checkpoint 已保存: %s (%d chunks)", checkpoint_path, len(all_chunks))

    # 嵌入 + 入库
    if all_chunks:
        # 断点续传: 检测 Milvus 中已有向量数, 自动跳过已入库段
        skip_chunks = 0
        if args.resume_from_checkpoint:
            store.connect()
            existing_count = store.count
            if existing_count > 0:
                skip_chunks = existing_count
                logger.info("Milvus 已有 %d 条向量, 跳过前 %d 条 (%.0f%%)",
                            existing_count, skip_chunks,
                            skip_chunks / len(all_chunks) * 100)
            store.disconnect()
        embed_and_insert(all_chunks, store, skip_chunks=skip_chunks)

    # 重建 BM25 索引 (全量用内存 chunks / 增量查 Milvus)
    logger.info("重建 BM25 索引...")
    if full_rebuild:
        store.build_bm25(
            [c.text for c in all_chunks],
            [c.doc_id for c in all_chunks],
            [c.spec_number for c in all_chunks],
            [c.chunk_index for c in all_chunks],
        )
        logger.info("BM25 索引已重建 (%d 条)", len(all_chunks))
    elif stats.get("new", 0) > 0 or stats.get("replaced", 0) > 0:
        n_bm25 = store.rebuild_bm25_from_collection()
        logger.info("BM25 索引已刷新 (%d 条)", n_bm25)
    else:
        logger.info("BM25 索引无需更新")

    # 统计
    total_elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("摄入完成 (模式: %s)", "全量重建" if full_rebuild else "增量")
    logger.info("  DOCX 总数: %d", stats["total"])
    logger.info("  新增: %d  替换: %d  跳过: %d  失败: %d", stats["new"], stats["replaced"], stats["skipped"], stats["failed"])
    logger.info("  Chunks: %d", stats["chunks"])
    logger.info("  总耗时: %.1fs (%.2f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 60)

    store.disconnect()


if __name__ == "__main__":
    main()
