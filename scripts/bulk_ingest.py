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

from src.config import settings, ingestion_config  # noqa: E402
from src.ingestion.embedder import BatchEmbedder  # noqa: E402
from src.ingestion.embedding_cache import EmbeddingCache  # noqa: E402
from src.ingestion.extractor import DoclingExtractor  # noqa: E402
from src.ingestion.manifest import (  # noqa: E402
    IngestionManifest,
    compare_versions,
    parse_3gpp_version,
)
from src.ingestion.mps_embedder import MPSChunkedEmbedder  # noqa: E402
from src.ingestion.splitter import HeaderAwareSplitter, classify_chunk  # noqa: E402
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
    O-RAN 格式:
      "O-RAN.WG1.CUS.0-R003-v11.00" → ("O-RAN.WG1.CUS.0", "v11.00")
    """
    # O-RAN 文件检测
    if stem.upper().startswith("O-RAN"):
        from src.ingestion.extractor import DoclingExtractor
        spec_id, release, version = DoclingExtractor.parse_oran_filename(stem + ".docx")
        return (spec_id, version) if spec_id else ("", version if version else None)
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


def find_docx_files(doc_dir: str, series_filter: list[str] | None = None) -> list[Path]:
    """收集统一文档目录下所有 DOCX 文件，按 Series 排序.

    目录结构: data/documents/R18/{21_series/,22_series/,...}
              data/documents/ORAN/
    跳过封面页: _cover.docx 仅含标题/版本信息，无规范正文。

    series_filter: 只包含指定 series 目录 (如 ['36','38']), None=全部.
                   ORAN 文档不受 series_filter 影响, 始终包含。
    """
    root = Path(doc_dir)
    if not root.exists():
        logger.error("目录不存在: %s", root)
        return []
    # 3GPP docs: data/documents/R18/...
    files_3gpp = sorted(root.rglob("*.docx"), key=lambda p: (p.parent.name, p.name))
    if series_filter:
        valid_dirs = {f"{s}_series" for s in series_filter}
        files_3gpp = [
            f for f in files_3gpp
            if any(d.name in valid_dirs for d in f.parents)
        ]
        logger.info(
            "Series 过滤: %s → 匹配 %d 个 3GPP 文件",
            series_filter, len(files_3gpp),
        )
    # ORAN docs: 同时扫描 ORAN 子目录 (优先 data/documents/ORAN, 其次 DOCUMENTS_DIR/ORAN)
    files_oran: list[Path] = []
    oran_dirs = [root.parent / "ORAN", root / "ORAN"]
    for oran_dir in oran_dirs:
        if oran_dir.exists():
            found = sorted(oran_dir.rglob("*.docx"), key=lambda p: p.name)
            files_oran.extend(found)
            logger.info("发现 ORAN 文档目录: %s (%d 文件)", oran_dir, len(found))
            break  # 只取第一个存在的目录
    all_files = files_3gpp + files_oran
    # 跳过仅封面页（无正文内容）
    all_files = [f for f in all_files if "_cover" not in f.stem.lower() and "cover" != f.stem.lower()]
    return all_files


# ── 单篇处理 ──

def _normalize_chunk_sizes(all_chunks: list, splitter: HeaderAwareSplitter) -> list:
    """拆分超字节上限的 chunk, 确保 checkpoint 中所有 chunk ≤ 55KB.

    在提取阶段调用, 保证保存到 checkpoint 的数据已是干净尺寸。
    避免 embed_and_insert 阶段因 chunk 数变化导致 skip_chunks 错位。
    """
    MAX_BYTES = 55000
    oversized = sum(1 for c in all_chunks if len(c.text.encode("utf-8")) > MAX_BYTES)
    if not oversized:
        return all_chunks

    logger.info("提取后规范化: 修正 %d 个超限 chunk...", oversized)
    safe: list = []
    for c in all_chunks:
        if len(c.text.encode("utf-8")) <= MAX_BYTES:
            safe.append(c)
        else:
            sub = splitter._fit_byte_limit(
                c.text, c.doc_id, c.series,
                c.spec_number, c.release,
                c.parent_section_id, c.parent_title,
                c.section_number, c.section_title, c.section_path,
                c.chunk_index,
            )
            safe.extend(sub)
    logger.info(
        "chunk 规范化完成: %d → %d (+%d), 最大 %dB",
        len(all_chunks), len(safe),
        len(safe) - len(all_chunks),
        max(len(c.text.encode("utf-8")) for c in safe),
    )

    # ── 最后兜底: 对仍超限的 chunk 做语义截断 (极少数, 丢失尾部 ~1-3%) ──
    # _fit_byte_limit 中的 line-split 在极端合并单元格下可能残留微超
    from src.retriever.milvus_store import _safe_truncate_bytes

    hard_capped = 0
    for c in safe:
        bs = len(c.text.encode("utf-8"))
        if bs > MAX_BYTES:
            c.text = _safe_truncate_bytes(c.text, MAX_BYTES)
            hard_capped += 1
            logger.warning(
                "  硬截断 %s chunk#%d: %dB → %dB (丢失 ~%.0f%%)",
                c.doc_id, c.chunk_index, bs,
                len(c.text.encode("utf-8")),
                100 * (bs - len(c.text.encode("utf-8"))) / bs,
            )
    if hard_capped:
        logger.info("硬截断完成: %d 个 chunk 安全截断至 ≤ %dB", hard_capped, MAX_BYTES)

    return safe


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

    # 判断文档类型
    is_oran = docx.stem.upper().startswith("O-RAN") or "ORAN" in str(docx.parent).upper()
    doc_type = "oran" if is_oran else "3gpp"

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
        if is_oran:
            # ORAN: 从文件名解析的 version 作为 release (如 v11.00)
            release = version if version else "ORAN"
        else:
            logger.warning("无法检测 Release, 默认 R18: %s", docx.name)
            release = "R18"

    if not result.markdown:
        return [], spec_number, release, version, sha

    doc_meta = {
        "doc_id": docx.stem,
        "series": int(spec_number.split(".")[0]) if spec_number and not is_oran else 0,
        "spec_number": spec_number,
        "release": release,
        "doc_type": doc_type,
    }
    chunks = splitter.split_document(result.markdown, doc_meta)
    # Annotate each chunk with rule-based metadata (content_type, spec_role, topic_domain)
    for chunk in chunks:
        meta = classify_chunk(chunk.text, spec_number, chunk.parent_title)
        chunk.content_type = meta["content_type"]
        chunk.spec_role = meta["spec_role"]
        chunk.topic_domain = meta["topic_domain"]
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
    """MPS 分段嵌入 + 逐段入库 Milvus (MilvusStore.insert 内部微批次防 gRPC 超限).

    每段嵌入后立即入库 (段失败不影响已入库段).
    """
    if not all_chunks:
        logger.warning("无 chunk 可入库")
        return 0

    store.connect()

    # ── 入库前安全校验: 检测超字节上限的 chunk ──
    # 不应出现 (提取阶段应已完成拆分); 若出现则发出明确警告
    MAX_CHUNK_BYTES = 55000
    oversized = sum(1 for c in all_chunks if len(c.text.encode("utf-8")) > MAX_CHUNK_BYTES)
    if oversized:
        logger.warning(
            "检测到 %d 个 chunk 超过 %d 字节安全上限! "
            "这通常是使用旧 checkpoint 数据所致. "
            "入库时 _safe_truncate_bytes 会在语义边界截断, 不会崩溃, "
            "但会丢失少量尾部内容. 建议重新运行提取阶段以获取完整数据.",
            oversized, MAX_CHUNK_BYTES,
        )

    total = len(all_chunks)
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
        seg_texts = [
            f"{c.section_title} {c.section_path} {c.text[:500]}" if c.section_path else c.text
            for c in all_chunks[start:end]
        ]
        seg_count = len(seg_texts)
        seg_id = seg_idx + 1

        logger.info("[段 %d/%d] 编码 %d 条文本...", seg_id, num_segments, seg_count)
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
            seg_id, num_segments, seg_count, seg_elapsed,
            seg_count / seg_elapsed if seg_elapsed > 0 else 0,
        )

        # 嵌入结果赋给 chunk (供 insert 使用)
        seg_chunks = all_chunks[start:end]
        for c, emb in zip(seg_chunks, seg_emb):
            c.embedding = emb

        # 逐段入库 (MilvusStore.insert 内部按 MAX_INSERT_BATCH=1000 微批次, 防 gRPC 超限)
        n = store.insert(seg_chunks)
        inserted_total += n
        logger.info(
            "[段 %d/%d] 入库: %d chunks → Milvus (累计 %d/%d)",
            seg_id, num_segments, n, inserted_total, total,
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
        "--series",
        nargs="*",
        default=None,
        help="只摄入指定 series 的文档 (如 --series 36 38), 默认全部; ORAN 始终包含",
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

    series_filter = args.series if args.series is not None and len(args.series) > 0 else None
    docx_files = find_docx_files(doc_dir, series_filter=series_filter)
    logger.info("找到 %d 个 DOCX 文件", len(docx_files))

    if not docx_files:
        logger.error("无 DOCX 文件")
        return

    extractor = DoclingExtractor()
    splitter = HeaderAwareSplitter(
        max_chunk_chars=ingestion_config.chunk_size,
        chunk_overlap_chars=ingestion_config.chunk_overlap,
        max_chunk_bytes=55000,
        chunk_mode=ingestion_config.chunk_mode,  # type: ignore[arg-type]
        table_max_chars=ingestion_config.table_max_chars,
        prose_max_chars=ingestion_config.prose_max_chars,
        max_chunk_hard_chars=ingestion_config.max_chunk_chars,
    )
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

        # 提取后规范化 chunk 大小 (零超限, 确保 checkpoint 干净)
        all_chunks = _normalize_chunk_sizes(all_chunks, splitter)

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

    # ── 构建离线 Cross-Reference Graph ──
    if full_rebuild or stats.get("new", 0) > 0 or stats.get("replaced", 0) > 0:
        logger.info("构建离线交叉引用图...")
        try:
            from src.ingestion.xref_graph_builder import XrefGraphBuilder

            store.connect()
            store.create_collection(drop_existing=False)
            builder = XrefGraphBuilder(store, target_series={38})
            xref_path = settings.data_abs_dir / "processed" / "xref_graph.json"
            builder.build(xref_path)
            logger.info("Xref Graph 已保存: %s", xref_path)
        except Exception as e:
            logger.warning("Xref Graph 构建失败 (不影响摄入主流程): %s", e)

    store.disconnect()


if __name__ == "__main__":
    main()
