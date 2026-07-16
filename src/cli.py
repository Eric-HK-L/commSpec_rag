"""命令行工具 — 数据摄入 / 索引管理.

用法:
    python -m src.cli stats            # 查看向量库统计
    python -m src.cli ingest --release R18 --series 38  # 全流程摄入
    python -m src.cli ingest --spec 38300 --dry-run      # 预览摄入计划
    python -m src.cli incremental                         # 增量索引

.. deprecated:: 2026-07-14
    `import` 子命令已废弃 (PrecomputedLoader)。请使用 `scripts/bulk_ingest.py`。
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("cli")


def cmd_import(args: argparse.Namespace) -> None:
    """[已废弃] PrecomputedLoader 已于 2026-07-14 废弃.

    请使用 `python scripts/bulk_ingest.py` 替代.
    """
    logger.error(
        "`import` 子命令已废弃。HuggingFace 预计算数据集不再使用。\n"
        "请使用: python scripts/bulk_ingest.py [--resume-from-checkpoint]"
    )
    sys.exit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    """查看 Milvus 向量库统计."""
    from src.retriever.milvus_store import MilvusStore

    store = MilvusStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection_name,
    )
    store.connect()
    logger.info("向量库: %s", store.__class__.__name__)
    logger.info("记录数: %d", store.count)
    logger.info("BM25: %s", "\u2713" if store.supports_bm25 else "\u2717")
    store.disconnect()


def cmd_ingest(args: argparse.Namespace) -> None:
    """Phase 2: DOCX 全流程摄入 (下载→转换→分块→嵌入→入库)."""
    from src.ingestion.orchestrator import IngestionOrchestrator
    from src.retriever.milvus_store import MilvusStore

    store = MilvusStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection_name,
    )
    store.connect()
    store.create_collection(drop_existing=False)

    def on_step(step: str, status: str) -> None:
        emoji = {"download": "📥", "extract": "📄", "split": "✂️", "embed": "🧮", "insert": "💾"}
        logger.info("%s %s: %s", emoji.get(step, "▶️"), step, status)

    orchestrator = IngestionOrchestrator(
        vector_store=store,
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
        skip_split=args.skip_split,
        skip_embed=args.skip_embed,
        on_step=on_step,
    )

    if args.dry_run:
        logger.info("[DRY-RUN] 预览摄入计划 (release=%s, series=%s, spec=%s)",
                     args.release, args.series, args.spec)
        logger.info("[DRY-RUN] 步骤: 下载 → Docling 转换 → 分块 → 嵌入 BGE/API → 入库 %s",
                     store.__class__.__name__)
        return

    stats = orchestrator.run_full_pipeline(
        release=args.release,
        series=args.series,
        spec=args.spec,
        docx_dir=args.docx_dir,
    )

    logger.info("摄入完成: %d/%d docs, %d chunks, %.1fs",
                 stats.docs_success, stats.docs_total,
                 stats.chunks_inserted, stats.elapsed_seconds)
    if stats.errors:
        for e in stats.errors:
            logger.warning("  ⚠️ %s", e)

    store.disconnect()


def cmd_feedback(args: argparse.Namespace) -> None:
    """反馈分析报告."""
    from src.generator.feedback import generate_report, get_stats

    if args.subcommand == "stats":
        import json as _json
        stats = get_stats()
        print(_json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.subcommand == "report":
        report = generate_report()
        print(report)


def cmd_incremental(args: argparse.Namespace) -> None:
    """Phase 2: 增量索引 — 仅处理变更文档."""
    from src.ingestion.incremental import IncrementalIndexer
    from src.retriever.milvus_store import MilvusStore

    store = MilvusStore(
        host=settings.milvus_host,
        port=settings.milvus_port,
        collection_name=settings.milvus_collection_name,
    )
    store.connect()
    indexer = IncrementalIndexer(store)
    indexer.load_state()

    new_files, modified_files, deleted_files = indexer.detect_changes()

    if args.dry_run:
        logger.info("[DRY-RUN] 变更检测: +%d ~%d -%d", len(new_files), len(modified_files), len(deleted_files))
        for f in new_files:
            logger.info("  + %s", f)
        for f in modified_files:
            logger.info("  ~ %s", f)
        for f in deleted_files:
            logger.info("  - %s", f)
        return

    if not any([new_files, modified_files, deleted_files]):
        logger.info("无变更")
        return

    stats = indexer.process_incremental(new_files, modified_files, deleted_files)
    logger.info("增量完成: +%d -%d chunks", stats["inserted"], stats["deleted"])
    store.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="3GPP RAG 管理工具")
    sub = parser.add_subparsers(dest="command")

    p_import = sub.add_parser("import", help="[已废弃] 导入预计算数据 — 请用 bulk_ingest.py")
    p_import.add_argument("--drop", action="store_true", help="(已废弃)")
    p_import.set_defaults(func=cmd_import)

    p_stats = sub.add_parser("stats", help="查看向量库统计")
    p_stats.set_defaults(func=cmd_stats)

    # Phase 2: DOCX 摄入
    p_ingest = sub.add_parser("ingest", help="DOCX 全流程摄入 (下载→转换→分块→嵌入→入库)")
    p_ingest.add_argument("--release", default="R18", help="3GPP Release (默认: R18)")
    p_ingest.add_argument("--series", type=int, help="Series 编号 (如 38)")
    p_ingest.add_argument("--spec", help="单篇规范号 (如 38300)")
    p_ingest.add_argument("--docx-dir", help="直接指定 DOCX 目录 (跳过下载)")
    p_ingest.add_argument("--skip-download", action="store_true", help="跳过下载步骤")
    p_ingest.add_argument("--skip-extract", action="store_true", help="跳过提取步骤")
    p_ingest.add_argument("--skip-split", action="store_true", help="跳过分块步骤")
    p_ingest.add_argument("--skip-embed", action="store_true", help="跳过嵌入步骤")
    p_ingest.add_argument("--dry-run", action="store_true", help="仅预览, 不执行")
    p_ingest.set_defaults(func=cmd_ingest)

    # Phase 2: 增量索引
    p_incr = sub.add_parser("incremental", help="增量索引 (仅处理变更文档)")
    p_incr.add_argument("--dry-run", action="store_true", help="仅预览变更")
    p_incr.set_defaults(func=cmd_incremental)

    # Phase 4: 反馈分析
    p_feedback = sub.add_parser("feedback", help="用户反馈统计与分析报告")
    p_feed_sub = p_feedback.add_subparsers(dest="subcommand")
    p_feed_stats = p_feed_sub.add_parser("stats", help="简要统计 (JSON)")
    p_feed_stats.set_defaults(func=cmd_feedback)
    p_feed_report = p_feed_sub.add_parser("report", help="生成 Markdown 分析报告")
    p_feed_report.set_defaults(func=cmd_feedback)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
