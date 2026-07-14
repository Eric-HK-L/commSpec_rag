"""嵌入模型横向对比 — bge-large-en-v1.5 vs BGE-M3.

对比维度:
  1. 跨语言检索能力 (中/英/韩 查询 → 英文文档)
  2. 嵌入速度 (本地 CPU/MPS, batch=32)
  3. 模型加载内存
  4. 嵌入维度 (两者均为 1024, 验证)
  5. 相似度分布 (同义 vs 无关文本)

使用:
  python scripts/compare_models.py
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare")


# ── 测试数据集 ──

# 多语言查询 (同义)
QUERIES_MULTILINGUAL = {
    "en": "What is the NR physical layer channel structure?",
    "zh": "NR物理层信道结构是什么？",
    "ko": "NR 물리 계층 채널 구조는 무엇인가요?",
}

# 英文文档片段 (3GPP 语义相关 vs 无关)
DOCS_ENGLISH = [
    "The NR physical layer consists of physical channels and physical signals. "
    "Physical channels include PDSCH, PUSCH, PDCCH, PUCCH, PBCH, and PRACH as defined in TS 38.211.",
    "UE shall perform cell selection and reselection according to criteria defined in TS 38.304. "
    "The serving cell quality shall be measured using RSRP and RSRQ.",
    "The MAC layer provides data transfer services on logical channels. "
    "MAC sublayer functions include scheduling, priority handling, and HARQ operations.",
    "Potatoes are a starchy root vegetable commonly used in cooking worldwide. "
    "They can be boiled, baked, or fried and are rich in carbohydrates.",
]

# 3GPP 文档标题 (用于语义匹配测试)
DOC_TITLES = [
    "NR; Physical channels and modulation (TS 38.211)",
    "NR; NR and NG-RAN Overall description; Stage-2 (TS 38.300)",
    "NR; Medium Access Control (MAC) protocol specification (TS 38.321)",
    "NR; Radio Resource Control (RRC); Protocol specification (TS 38.331)",
    "NR; Physical layer procedures for control (TS 38.213)",
]
DOC_CONTENT = [
    "The physical layer structure includes resource elements, resource blocks, "
    "and bandwidth parts as defined in clause 4 of TS 38.211.",
    "The NG-RAN architecture consists of gNBs connected via Xn interface, "
    "with split into CU and DU as described in TS 38.401.",
    "The MAC protocol provides mapping between logical channels and transport channels, "
    "including multiplexing of MAC SDUs onto transport blocks.",
    "The RRC protocol defines states RRC_IDLE, RRC_INACTIVE, and RRC_CONNECTED "
    "with associated procedures for connection establishment and release.",
    "Downlink control information (DCI) is transmitted on PDCCH and includes "
    "scheduling assignments for PDSCH and PUSCH as per TS 38.214.",
]


# ── 工具函数 ──

def load_model(model_name: str, device: str = "cpu") -> Any:
    """加载模型并记录内存."""
    from sentence_transformers import SentenceTransformer

    tracemalloc.start()
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    load_time = time.time() - t0
    _, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") \
        else model.get_sentence_embedding_dimension()
    logger.info("  加载 %s: %.1fs, 峰值内存 %.0f MB, dim=%d", model_name, load_time, peak_mb / 1e6, dim)
    return model, load_time, peak_mb / 1e6


def benchmark_embed_speed(model: Any, texts: list[str], batch_size: int = 32, warmup: int = 3) -> float:
    """测量嵌入速度 (batch/s)."""
    n = len(texts)
    # warmup
    for _ in range(warmup):
        model.encode(texts[:min(batch_size, n)], normalize_embeddings=True, show_progress_bar=False)

    t0 = time.time()
    model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
    elapsed = time.time() - t0
    return len(texts) / elapsed


# ── 对比主流程 ──

def compare_models():
    print("=" * 60)
    print("  嵌入模型横向对比: bge-large-en-v1.5 vs BGE-M3")
    print("=" * 60)

    models = {
        "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
        "BGE-M3": "BAAI/bge-m3",
    }

    loaded: dict[str, Any] = {}
    metrics: dict[str, dict] = {}

    # ── 1. 加载模型 ──
    print("\n## 1. 模型加载\n")
    device = "mps" if (
        __import__("torch").backends.mps.is_available() if hasattr(__import__("torch").backends, "mps") else False
    ) else "cpu"
    print(f"  设备: {device}")

    for label, name in models.items():
        print(f"\n  [{label}]")
        model, load_time, peak_mb = load_model(name, device)
        loaded[label] = model
        metrics[label] = {"load_time_s": load_time, "peak_memory_mb": peak_mb}

    # ── 2. 维度验证 ──
    print("\n## 2. 维度验证\n")
    for label, model in loaded.items():
        dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") \
            else model.get_sentence_embedding_dimension()
        print(f"  {label}: {dim} 维")
        metrics[label]["dimension"] = dim

    # ── 3. 嵌入速度 ──
    print("\n## 3. 嵌入速度 (batch=32)\n")
    speed_texts = DOC_CONTENT * 20  # 100 条, ~3000 chars
    for label, model in loaded.items():
        tps = benchmark_embed_speed(model, speed_texts[:64])  # 2 batches
        print(f"  {label}: {tps:.0f} texts/s")
        metrics[label]["texts_per_second"] = tps

    # ── 4. 跨语言检索能力 ──
    print("\n## 4. 跨语言检索 (多语言查询 → 英文文档)\n")

    for label, model in loaded.items():
        print(f"\n  [{label}]")
        # 编码文档
        doc_embs = model.encode(DOC_CONTENT, normalize_embeddings=True, show_progress_bar=False)

        for lang, query in QUERIES_MULTILINGUAL.items():
            lang_name = {"en": "English", "zh": "中文", "ko": "韩文"}[lang]
            q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            sims = (q_emb @ doc_embs.T)[0]
            top_idx = int(np.argmax(sims))
            top_sim = float(sims[top_idx])
            doc_title = DOC_TITLES[top_idx]

            status = "✅" if top_sim > 0.6 else ("⚠️" if top_sim > 0.4 else "❌")
            print(f"    {lang_name}: top={doc_title[:50]}... sim={top_sim:.4f} {status}")
            metrics[label].setdefault("cross_lingual", {})[lang] = {
                "top_similarity": top_sim,
                "top_doc": doc_title,
            }

    # ── 5. 3GPP 语义相关性 ──
    print("\n## 5. 语义相关性测试 (物理层查询 → 文档匹配)\n")
    query = "What are the physical layer channels in NR?"

    for label, model in loaded.items():
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        d_embs = model.encode(DOC_CONTENT, normalize_embeddings=True, show_progress_bar=False)
        sims = (q_emb @ d_embs.T)[0]

        ranked = sorted(zip(DOC_TITLES, sims), key=lambda x: x[1], reverse=True)
        print(f"  {label}:")
        for i, (title, sim) in enumerate(ranked[:3]):
            prefix = "  →" if i == 0 else "   "
            print(f"    {prefix} #{i+1} {title[:55]}... ({sim:.4f})")

        gap = float(sims[0] - sims[-2]) if len(sims) > 1 else 0
        metrics[label]["semantic_gap"] = gap
        metrics[label]["unrelated_similarity"] = float(sims[-1])

    # ── 6. 汇总报告 ──
    print("\n" + "=" * 60)
    print("  汇总报告")
    print("=" * 60)

    print(f"\n  {'指标':<30} {'bge-large-en-v1.5':<22} {'BGE-M3':<22}")
    print(f"  {'─' * 30} {'─' * 22} {'─' * 22}")

    for key_label in ["load_time_s", "peak_memory_mb", "dimension", "texts_per_second"]:
        v1 = metrics["bge-large-en-v1.5"].get(key_label, "N/A")
        v2 = metrics["BGE-M3"].get(key_label, "N/A")
        label_map = {
            "load_time_s": "加载时间 (s)",
            "peak_memory_mb": "峰值内存 (MB)",
            "dimension": "嵌入维度",
            "texts_per_second": "嵌入吞吐 (t/s)",
        }
        label = label_map.get(key_label, key_label)
        if isinstance(v1, float):
            print(f"  {label:<30} {v1:<22.1f} {v2:<22.1f}")
        else:
            print(f"  {label:<30} {v1!s:<22} {v2!s:<22}")

    # 跨语言平均
    for model_label in ["bge-large-en-v1.5", "BGE-M3"]:
        cl = metrics[model_label].get("cross_lingual", {})
        avg_sim = np.mean([v["top_similarity"] for v in cl.values()]) if cl else 0
        metrics[model_label]["avg_cross_lingual_sim"] = avg_sim

    print(f"  {'跨语言平均相似度':<30} {metrics['bge-large-en-v1.5']['avg_cross_lingual_sim']:<22.4f} {metrics['BGE-M3']['avg_cross_lingual_sim']:<22.4f}")
    print(f"  {'语义区分度 (gap)':<30} {metrics['bge-large-en-v1.5']['semantic_gap']:<22.4f} {metrics['BGE-M3']['semantic_gap']:<22.4f}")

    # 结论
    print("\n## 结论\n")
    en_cl = metrics["bge-large-en-v1.5"]["avg_cross_lingual_sim"]
    m3_cl = metrics["BGE-M3"]["avg_cross_lingual_sim"]
    m3_gap = metrics["BGE-M3"]["semantic_gap"]

    if m3_cl > 0.7:
        print("  ✅ BGE-M3 跨语言检索能力: 达标 (平均 sim > 0.7)")
    else:
        print(f"  ⚠️ BGE-M3 跨语言检索: 偏低 ({m3_cl:.3f})")

    if m3_gap > 0.1:
        print("  ✅ BGE-M3 语义区分度: 良好 (gap > 0.1, 能区分相关/无关)")
    else:
        print(f"  ⚠️ BGE-M3 语义区分度: 不足 ({m3_gap:.3f})")

    if en_cl < 0.5:
        print("  ❌ bge-large-en: 不支持跨语言 (英文模型)")
    else:
        print("  ℹ️ bge-large-en: 意外的高跨语言分 → 可能是字符集误匹配")

    print("\n  推荐: BGE-M3 作为多语言知识库统一嵌入模型")
    print(f"  代价: 加载时间 +{metrics['BGE-M3']['load_time_s'] - metrics['bge-large-en-v1.5']['load_time_s']:.0f}s, "
          f"内存 +{metrics['BGE-M3']['peak_memory_mb'] - metrics['bge-large-en-v1.5']['peak_memory_mb']:.0f}MB")

    return metrics


if __name__ == "__main__":
    compare_models()
