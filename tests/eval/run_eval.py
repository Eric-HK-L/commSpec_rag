"""评测执行脚本 — 加载测试集 → plan() 全链路检索 → 计算指标 → 输出报告.

评测走 RetrievalPlanner.plan() 全链路 (候选池=reranker_top_k=100,
Cross-Encoder 精排生效, 多跳/图扩展/taxonomy 分解/filter_noise 参与),
而非轻量 search() 路径 (20 候选无重排, 低估真实召回).

用法:
    python -m tests.eval.run_eval                          # 默认 test_set.json
    python -m tests.eval.run_eval --test-set my_set.json   # 自定义测试集
    python -m tests.eval.run_eval --fresh                  # 忽略 checkpoint 全量重算
    python -m tests.eval.run_eval --checkpoint /tmp/ck.json  # 自定义 checkpoint 路径
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .metrics import EvalReport, EvalResult, EvalSample, evaluate_batch, evaluate_one

# search() 轻量路径的历史基线 (2026-08 评测, 20 候选无重排, Recall@5=0.769)
# 仅作报告参照列 — 新评测结果与之对比, 定位 plan() 全链路的增益
LEGACY_SEARCH_RECALL_AT_5 = 0.769

CHECKPOINT_VERSION = 1


def load_test_set(path: str) -> list[EvalSample]:
    """从 JSON 文件加载测试集."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    samples: list[EvalSample] = []
    for item in data:
        samples.append(EvalSample(
            question=item["question"],
            expected_specs=item.get("expected_specs", []),
            expected_sections=item.get("expected_sections", []),
            difficulty=item.get("difficulty", "medium"),
            multi_hop=item.get("multi_hop", False),
        ))
    return samples


def format_report(report: EvalReport, elapsed_ms: float) -> str:
    """格式化评测报告为 Markdown 表格."""
    lines = [
        "# CommSpec RAG 检索评测报告",
        "",
        f"- **样本数**: {report.total}",
        f"- **耗时**: {elapsed_ms / 1000:.1f}s",
        "",
        "## 综合指标",
        "",
        "| 指标 | 值 | 目标 |",
        "|------|-----|------|",
        f"| Recall@5 (重排后) | {report.recall_at_5:.4f} | ≥ 0.80 |",
        f"| Recall@5 (初检) | {report.initial_recall_at_5:.4f} | 诊断参考 |",
        f"| Recall@5 (search路径参照) | {LEGACY_SEARCH_RECALL_AT_5:.4f} | legacy 基线 |",
        f"| Recall@10 | {report.recall_at_10:.4f} | ≥ 0.85 |",
        f"| Recall@20 | {report.recall_at_20:.4f} | ≥ 0.90 |",
        f"| 章节级 Recall@5 | {report.section_recall_at_5:.4f} | 参考 |",
        f"| 章节级 Recall@10 | {report.section_recall_at_10:.4f} | 参考 |",
        f"| MRR | {report.mrr:.4f} | ≥ 0.70 |",
        f"| NDCG@10 | {report.ndcg_at_10:.4f} | ≥ 0.75 |",
    ]

    if report.by_difficulty:
        lines.extend(["", "## 按难度", ""])
        for diff, metrics in report.by_difficulty.items():
            lines.append(f"### {diff} ({metrics['count']} 题)")
            lines.append("")
            lines.append(f"- Recall@5: {metrics['recall@5']:.4f}")
            lines.append(f"- 初检 Recall@5: {metrics.get('initial_recall@5', 0.0):.4f}")
            lines.append(f"- Recall@10: {metrics['recall@10']:.4f}")
            lines.append(f"- MRR: {metrics['mrr']:.4f}")
            lines.append(f"- NDCG@10: {metrics['ndcg@10']:.4f}")

    if report.by_multi_hop:
        lines.extend(["", "## 按检索类型", ""])
        for hop, metrics in report.by_multi_hop.items():
            label = "多跳检索" if hop else "单步检索"
            lines.append(f"### {label} ({metrics['count']} 题)")
            lines.append("")
            lines.append(f"- Recall@5: {metrics['recall@5']:.4f}")
            lines.append(f"- 初检 Recall@5: {metrics.get('initial_recall@5', 0.0):.4f}")
            lines.append(f"- Recall@10: {metrics['recall@10']:.4f}")
            lines.append(f"- MRR: {metrics['mrr']:.4f}")
            lines.append(f"- NDCG@10: {metrics['ndcg@10']:.4f}")

    return "\n".join(lines)


def _sample_key(sample: EvalSample) -> str:
    """样本唯一键 — question + expected_specs + expected_sections 的哈希.

    同一测试集内容变更 (如改期望规范) 会生成不同 key, 避免 checkpoint 误命中.
    """
    payload = json.dumps(
        [sample.question, sample.expected_specs, sample.expected_sections],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, dict]:
    """加载 checkpoint — 损坏/版本不符时返回空 dict, 触发重算."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == CHECKPOINT_VERSION:
            return data.get("samples", {})
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_checkpoint(path: Path, samples: dict[str, dict]) -> None:
    """原子写 checkpoint — 先写临时文件再 rename, 避免中断写坏."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"version": CHECKPOINT_VERSION, "samples": samples},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _extract_specs(results) -> list[str]:
    """从检索结果列表提取规范编号 (去空值)."""
    return [
        r.spec_number for r in results
        if hasattr(r, 'spec_number') and r.spec_number
    ]


def _extract_spec_sections(results) -> list[tuple[str, str]]:
    """从检索结果列表提取 [(spec_number, section_number), ...] 用于章节级召回."""
    pairs = []
    for r in results:
        if not hasattr(r, 'spec_number') or not r.spec_number:
            continue
        section = getattr(r, 'section_number', '') or getattr(r, 'parent_section_id', '') or ''
        pairs.append((r.spec_number, section))
    return pairs


def run_eval(
    test_set_path: str,
    dry_run: bool = False,
    checkpoint: str | None = None,
    fresh: bool = False,
) -> None:
    """执行评测主流程 — plan() 全链路检索 + checkpoint 断点续跑."""
    # 1. 加载测试集
    samples = load_test_set(test_set_path)
    print(f"✅ 加载测试集: {len(samples)} 条")

    # 2. 空跑模式: 仅验证测试集格式, 不连接 Milvus
    if dry_run:
        _dry_run(samples)
        return

    # 3. 初始化检索器 (RAGPipeline.plan: 完整全链路检索)
    try:
        from src.config import settings
        from src.generator.pipeline import RAGPipeline
        from src.retriever.milvus_store import MilvusStore

        store = MilvusStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection_name,
        )
        store.connect()
        pipeline = RAGPipeline(vector_store=store)
        print(f"✅ 检索器就绪 (Milvus: {store.count} chunks, 候选池={settings.reranker_top_k})")
    except Exception as e:
        print(f"⚠️ 检索器初始化失败 (Milvus 未运行?): {e}")
        print("进入空跑模式 — 仅验证测试集格式")
        _dry_run(samples)
        return

    # 4. checkpoint 加载 (断点续跑: 已算样本跳过)
    checkpoint_path = Path(checkpoint) if checkpoint else (
        Path(test_set_path).parent / "eval_checkpoint.json"
    )
    cached: dict[str, dict] = {} if fresh else _load_checkpoint(checkpoint_path)
    if cached:
        print(f"⏭️ checkpoint 命中: 跳过 {len(cached)} 条已计算样本")

    # 5. 逐条评测
    results: list[EvalResult] = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        key = _sample_key(sample)
        if key in cached:
            entry = cached[key]
            results.append(evaluate_one(
                sample,
                entry["retrieved_specs"],
                initial_specs=entry.get("initial_specs", []),
                retrieved_pairs=entry.get("retrieved_pairs", []),
            ))
            continue
        try:
            ctx = pipeline.plan(sample.question)
            retrieved_specs = _extract_specs(ctx.results)
            initial_specs = _extract_specs(getattr(ctx, "initial_results", []))
            retrieved_pairs = _extract_spec_sections(ctx.results)
            results.append(evaluate_one(
                sample, retrieved_specs, initial_specs=initial_specs,
                retrieved_pairs=retrieved_pairs,
            ))
            cached[key] = {
                "retrieved_specs": retrieved_specs,
                "initial_specs": initial_specs,
                "retrieved_pairs": retrieved_pairs,
            }
            _save_checkpoint(checkpoint_path, cached)
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(samples)}")
        except Exception as e:
            print(f"  ⚠️ 第 {i + 1} 条评测失败: {e}")
    elapsed = (time.time() - t0) * 1000

    # 6. 聚合报告
    report = evaluate_batch(results)
    output = format_report(report, elapsed)
    print("\n" + output)

    # 7. 保存报告
    report_path = Path(test_set_path).parent / "eval_report.md"
    report_path.write_text(output, encoding="utf-8")
    print(f"\n📄 报告已保存: {report_path}")
    print(f"💾 checkpoint 已保存: {checkpoint_path}")


def _dry_run(samples: list[EvalSample]) -> None:
    """空跑验证: 检查测试集格式完整性."""
    issues = 0
    for i, s in enumerate(samples):
        if not s.question:
            print(f"  ❌ 第 {i + 1} 条: question 为空")
            issues += 1
        if not s.expected_specs:
            print(f"  ⚠️ 第 {i + 1} 条: expected_specs 为空")
            issues += 1
    if issues == 0:
        print(f"✅ 测试集格式检查通过 ({len(samples)} 条)")
    else:
        print(f"⚠️ {issues} 处格式问题")
    print("\n提示: 确保 API 服务运行后再执行真实评测")
    print("  cd src && python -m api.main  # 启动 API")


def main():
    parser = argparse.ArgumentParser(description="CommSpec RAG 检索评测 (plan 全链路)")
    parser.add_argument(
        "--test-set", default=None,
        help="测试集 JSON 路径 (默认: tests/eval/test_set.json)"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint JSON 路径 (默认: <test-set 同目录>/eval_checkpoint.json)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="忽略已有 checkpoint, 全量重算"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅验证测试集格式, 不连接 Milvus (适用于 CI)"
    )
    args = parser.parse_args()

    test_set = args.test_set or str(
        Path(__file__).resolve().parent / "test_set.json"
    )
    run_eval(
        test_set,
        dry_run=args.dry_run,
        checkpoint=args.checkpoint,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
