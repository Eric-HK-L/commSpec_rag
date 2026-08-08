"""评测执行脚本 — 加载测试集 → 检索 → 计算指标 → 输出报告.

用法:
    python -m tests.eval.run_eval                          # 默认 test_set.json
    python -m tests.eval.run_eval --test-set my_set.json   # 自定义测试集
    python -m tests.eval.run_eval --top-k 20               # 指定检索 Top-K
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .metrics import EvalReport, EvalResult, EvalSample, evaluate_batch, evaluate_one


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
        f"| Recall@5 | {report.recall_at_5:.4f} | ≥ 0.80 |",
        f"| Recall@10 | {report.recall_at_10:.4f} | ≥ 0.85 |",
        f"| Recall@20 | {report.recall_at_20:.4f} | ≥ 0.90 |",
        f"| MRR | {report.mrr:.4f} | ≥ 0.70 |",
        f"| NDCG@10 | {report.ndcg_at_10:.4f} | ≥ 0.75 |",
    ]

    if report.by_difficulty:
        lines.extend(["", "## 按难度", ""])
        for diff, metrics in report.by_difficulty.items():
            lines.append(f"### {diff} ({metrics['count']} 题)")
            lines.append("")
            lines.append(f"- Recall@5: {metrics['recall@5']:.4f}")
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
            lines.append(f"- Recall@10: {metrics['recall@10']:.4f}")
            lines.append(f"- MRR: {metrics['mrr']:.4f}")
            lines.append(f"- NDCG@10: {metrics['ndcg@10']:.4f}")

    return "\n".join(lines)


def run_eval(test_set_path: str, top_k: int = 20, dry_run: bool = False) -> None:
    """执行评测主流程."""
    # 1. 加载测试集
    samples = load_test_set(test_set_path)
    print(f"✅ 加载测试集: {len(samples)} 条")

    # 2. 空跑模式: 仅验证测试集格式, 不连接 Milvus
    if dry_run:
        _dry_run(samples)
        return

    # 3. 初始化检索器 (完整 RAGPipeline: 查询扩展 + Dense+BM25 混合检索)
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
        print(f"✅ 检索器就绪 (Milvus: {store.count} chunks)")
    except Exception as e:
        print(f"⚠️ 检索器初始化失败 (Milvus 未运行?): {e}")
        print("进入空跑模式 — 仅验证测试集格式")
        _dry_run(samples)
        return

    # 4. 逐条评测
    results: list[EvalResult] = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        try:
            retrieval = pipeline.search(sample.question, top_k=top_k)
            retrieved_specs = [
                r.spec_number for r in retrieval
                if hasattr(r, 'spec_number') and r.spec_number
            ]
            result = evaluate_one(sample, retrieved_specs)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(samples)}")
        except Exception as e:
            print(f"  ⚠️ 第 {i + 1} 条评测失败: {e}")
    elapsed = (time.time() - t0) * 1000

    # 4. 聚合报告
    report = evaluate_batch(results)
    output = format_report(report, elapsed)
    print("\n" + output)

    # 5. 保存报告
    report_path = Path(test_set_path).parent / "eval_report.md"
    report_path.write_text(output, encoding="utf-8")
    print(f"\n📄 报告已保存: {report_path}")


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
    parser = argparse.ArgumentParser(description="CommSpec RAG 检索评测")
    parser.add_argument(
        "--test-set", default=None,
        help="测试集 JSON 路径 (默认: tests/eval/test_set.json)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="检索 Top-K (默认: 20)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅验证测试集格式, 不连接 Milvus (适用于 CI)"
    )
    args = parser.parse_args()

    test_set = args.test_set or str(
        Path(__file__).resolve().parent / "test_set.json"
    )
    run_eval(test_set, top_k=args.top_k, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
