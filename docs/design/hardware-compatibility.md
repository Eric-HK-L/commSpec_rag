# 硬件平台兼容性指南

> 本文档说明 3GPP RAG 系统在不同硬件平台上的运行策略、已知限制与推荐配置。
> 最后更新: 2026-07-13

---

## 1. 支持的平台总览

| 平台 | CPU | GPU | 嵌入设备 | 推荐场景 |
|---|---|---|---|---|
| **Linux x86_64 + NVIDIA GPU** | Intel/AMD x64 | CUDA ✅ | `cuda` | 生产环境首选 |
| **Linux aarch64 + NVIDIA GB10** | ARM (Grace Blackwell) | CUDA ✅ (Blackwell) | `cuda` | 公司 ARM AI 服务器 |
| **Linux aarch64 + NVIDIA Grace** | ARM (Grace) | CUDA ✅ | `cuda` | ARM 服务器 |
| **Linux x86_64 (无 GPU)** | Intel/AMD x64 | 无 | `cpu` | 低成本服务器 |
| **macOS Apple Silicon** | ARM (M1-M4) | MPS ✅ | `mps` (批量嵌入) / `cpu` (API服务) | 开发 + 批量嵌入 |
| **macOS Intel** | Intel x64 | 无 | `cpu` | 旧款 Mac |
| **Windows x86_64 + NVIDIA GPU** | Intel/AMD x64 | CUDA ✅ | `cuda` | Windows 生产 |
| **Windows x86_64 (无 GPU)** | Intel/AMD x64 | 无 | `cpu` | Windows 开发 |

---

## 2. 设备自动检测

系统启动时通过 `get_hardware_info()` 自动识别硬件平台，日志输出示例：

```
# NVIDIA GPU 服务器
[INFO] 硬件检测: Linux / Intel/AMD x64 / GPU=nvidia_cuda (24576MB)

# Apple Silicon Mac
[INFO] 硬件检测: macOS / Apple Silicon / GPU=apple_mps
[WARNING] 检测到 Apple MPS GPU — 注意: MPS 不支持 torch.mps.empty_cache(), ...

# 无 GPU 服务器
[INFO] 硬件检测: Linux / Intel/AMD x64 / GPU=none
```

`get_best_device()` 返回值:

| 检测到的 GPU | 返回值 | 原因 |
|---|---|---|
| NVIDIA CUDA | `"cuda"` | 全功能、无已知问题 |
| Apple MPS | `"cpu"` | 内存泄漏风险, 安全降级 |
| 无 GPU | `"cpu"` | 唯一选项 |

---

## 3. 嵌入设备配置

`.env` 中 `EMBEDDING_DEVICE` 支持:

```
EMBEDDING_DEVICE=auto    # 自动检测 (推荐，大部分场景)
EMBEDDING_DEVICE=cuda    # 强制 NVIDIA GPU
EMBEDDING_DEVICE=cpu     # 强制 CPU
EMBEDDING_DEVICE=mps     # 强制 Apple MPS (需接受内存风险)
```

---

## 4. Apple Silicon MPS 加速方案 (✅ 已验证)

### 4.1 MPS 内存模型与水位线

Apple M1/M2/M3/M4 芯片使用**统一内存架构** (UMA)，CPU 和 GPU 共享物理内存。
Metal 框架将 GPU 可访问内存标记为 **wired**（不可换页），仅在进程退出时归还 OS。

| 特性 | CUDA (NVIDIA) | MPS (Apple Silicon) |
|---|---|---|
| `empty_cache()` | ✅ 主动清空缓存池 | ✅ PyTorch 2.13+ 支持, 但**仅释放 PyTorch 内部池, Metal 不归还 wired** |
| `HIGH_WATERMARK_RATIO` | N/A | ✅ 有效 (默认 1.0 = 100% 系统内存) |
| `LOW_WATERMARK_RATIO` | N/A | ✅ 必须同时设置 (自动计算值 = HIGH×4, 需 ≤1.0) |
| GPU 内存回收 | 显存隔离, 可释放 | 统一内存, wired 仅在进程退出时回收 |

**关键发现**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 默认值 1.0 允许 MPS 占用 100% 系统内存 (64GB → 持满 57GB)，这是之前 "MPS 内存爆炸" 的根因。

### 4.2 水位线配置

```bash
# .env
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5  # 上限 ~25GB
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3   # 必须同时设置, 否则 auto=HIGH×4 超 1.0 报错
```

> 注意: 必须在 `import torch` **之前**设置, 否则不生效。`_mps_embed_subprocess.py` 已通过 `os.environ.setdefault()` 在模块顶部自动设置。

### 4.3 MPS 子进程隔离方案 (✅ 已实现)

**设计**: `bulk_ingest.py` → 启动独立 `_mps_embed_subprocess.py` 子进程 → 每段 10,000 文本 → 子进程完成后退 → OS 回收 wired 内存。

**为什么不用 fork / multiprocessing**:
- Apple: fork() 后使用 Metal/GPU 是 undefined behavior
- 父进程已 import pymilvus (gRPC FD), spawn 会连带污染
- MPS 多进程同时提交 Metal command buffer → GPU 调度器死锁
- 因此用 `subprocess.Popen` 干净隔离, 单 worker, 零 IPC

**子进程内部**:
- 每 `chunks_per_worker=500` 批后重载模型 → `del model + gc.collect + empty_cache + sync`
- 每批次后 `torch.mps.empty_cache()`, 每 10 批 `gc.collect + mps.synchronize`
- 模型仅加载时接入 MPS, `torch.no_grad()` 禁用梯度

### 4.4 Batch Size 与 O(n²) 注意力惩罚 (⚠️ 核心发现)

**问题**: MPS 网上资料普遍建议 "大批次更快"，但实测 BGE-M3 相反。

**根因**: BGE-M3 (XLM-RoBERTa) 的 `scaled_dot_product_attention` 计算复杂度为 **O(n²)**，
其中 n = batch_size × max_seq_len。同时，SentenceTransformer 的 `encode()` 会将
a batch 内所有文本 pad 到该 batch 最长文本的长度，大批次中一条长文本拖累全组。

**实测数据** (M4 Max 40核 GPU, 1000 条相同文本, avg=1708 chars, ratio=0.5):

| batch | batches | n (tokens) | O(n²) | 每批耗时 | 总耗时 | t/s | wired |
|---|---|---|---|---|---|---|
| 4 | 250 | 2048 | 4.2M | 0.25s | **63s** | **15.9** | 8GB |
| 8 | 125 | 4096 | 16.8M | 0.62s | 77s | 13.1 | 10GB |
| 16 | 63 | 8192 | 67M | 1.70s | 107s | 9.3 | 16GB |
| 24 | 42 | 12288 | 151M | 4.33s | 182s | 5.5 | 9GB |
| 32 | 31 | 16384 | 268M | 7.13s | 228s | 4.4 | 30GB |

每批耗时完美追踪 O(n²): batch 翻倍 → 4× 时间。padding 放大效应使大批次实际损失更大。

> 此结论经三次独立网上验证确认: [Dify Rerank 性能调优](https://wenku.csdn.net/column/2vvkrrs120)、
> [长文本精调指南](https://zhuanlan.zhihu.com/p/696942462)、
> Sentence-BERT 原论文 (smart batching 策略)。

### 4.5 最优配置

| 参数 | 值 | 说明 |
|---|---|---|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | 0.5 | ~25GB 上限, 够用不影响速度 |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | 0.3 | 必须同时设置 |
| `--batch-size` | **4** | BGE-M3 长文本场景最优 |
| `--chunks-per-worker` | 500 | 每 500 批重载模型回收缓存 |
| `EMBEDDING_DEVICE` | mps | .env 设置 |

**108,948 文本全量预估**: ~1.9 小时, wired 峰值 ~8GB, 零 swap。

> ratio 本身不影响速度——只在极端值 (太高→swap, 太低→频繁GC) 时才有影响, 0.5 对 batch=4 完全透明。

---

## 5. NVIDIA GPU (CUDA)

### 5.1 环境要求

- NVIDIA 驱动程序 ≥ 525
- CUDA Toolkit ≥ 12.1
- PyTorch CUDA 版本 (`pip install torch --index-url https://download.pytorch.org/whl/cu121`)

### 5.2 推荐配置

```bash
# .env
EMBEDDING_DEVICE=cuda    # 或 auto (自动检测 CUDA)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

### 5.3 性能参考

| GPU | 显存 | 嵌入速度 (BGE-M3, batch=64) | 108K chunks 预估 |
|---|---|---|---|
| NVIDIA A100 80GB | 80 GB | ~0.02s/batch | ~2.2 分钟 |
| NVIDIA RTX 4090 | 24 GB | ~0.04s/batch | ~4.5 分钟 |
| NVIDIA T4 | 16 GB | ~0.10s/batch | ~11 分钟 |
| NVIDIA GTX 1080 | 8 GB | ~0.15s/batch | ~17 分钟 |

> **注**: 以上为估算值, 实际速度受 PCIe 带宽、CPU 主频、磁盘 I/O 影响。

---

## 6. NVIDIA Grace (ARM + CUDA)

NVIDIA Grace 是 ARM 架构 CPU, 搭配 NVIDIA GPU (如 Grace Hopper Superchip 自带 H100)。

### 6.1 环境要求

- Linux aarch64
- PyTorch ARM + CUDA 构建
- 通过 `get_hardware_info()` 自动识别为 "NVIDIA Grace"

### 6.2 特殊性

- PyTorch 需要 ARM64 + CUDA 版本 (非标准 x86_64 wheel)
- CPU 架构为 aarch64, `platform.machine()` 返回 `"aarch64"`, 检测逻辑通过内核 `processor` 字段中的 "grace" 标识区分

---

## 7. Intel/AMD x64 CPU (无 GPU)

### 7.1 推荐配置

```bash
EMBEDDING_DEVICE=cpu
EMBEDDING_PROVIDER=api    # 推荐云端 API, 避免本地嵌入瓶颈
# EMBEDDING_MODEL=text-embedding-3-small  # API 模式时取消注释
```

### 7.2 优化建议

| 优化项 | 配置 | 说明 |
|---|---|---|
| 批大小 | `--batch-size 64` | 增大可提升吞吐 (受内存限制) |
| 嵌入 API | `EMBEDDING_PROVIDER=api` | 比本地 BGE 快 10-100× |
| Worker 数 | `API_WORKERS=1` | 避免多进程各加载一份模型 |

### 7.3 性能参考 (BGE-M3, batch=64)

| CPU | 嵌入速度 | 108K chunks 预估 |
|---|---|---|
| AMD EPYC 9654 (96核) | ~0.5s/batch | ~28 分钟 |
| Intel Xeon 8480+ (56核) | ~0.8s/batch | ~45 分钟 |
| Intel Core i9-13900K | ~1.2s/batch | ~68 分钟 |
| Apple M4 Max (CPU) | ~4s/batch | ~3.8 小时 |
| Intel Core i7-1185G7 | ~6s/batch | ~5.7 小时 |

---

## 8. 内存策略

### 8.1 内存需求估算

| 组件 | BGE-M3 占用 |
|---|---|
| 嵌入模型 | ~2.2 GB |
| PyTorch 运行时 | ~0.4 GB |
| 每个 batch (32 chunks) | ~0.6 GB |
| 嵌入缓存 (SQLite) | ~0.2 GB |
| Milvus Standalone | ~2 GB |
| **推荐最小内存** | **10 GB** |
| **生产推荐** | **20 GB+** |

### 8.2 当前环境限制

- 本项目运行环境硬上限: **35 GB** (参见 `运行内存上限为35GB` 配置记录)
- 超过此阈值应终止进程并排查 (实测 MPS 泄漏达 57GB)

---

## 9. 嵌入模型替代方案

对无 GPU 或低内存环境, 可选用更小的嵌入模型:

| 模型 | 维度 | 大小 | 多语言 | 速度 (相对) | 适用场景 |
|---|---|---|---|---|---|
| **BAAI/bge-m3** | 1024 | 2.2 GB | ✅ 100+ 语言 | 1× | **多语言知识库首选**, 含稀疏向量输出 |
| BAAI/bge-base-en-v1.5 | 768 | 0.44 GB | ❌ 仅英文 | 4× | 中等精度, CPU (纯英文备选) |
| text-embedding-3-small (API) | 1536 | N/A | ✅ | 100×+ | 无本地 GPU |

> **BGE-M3 推荐理由**: 同一向量空间支持中/英/韩等多语言混合文档 + 跨语言检索, 无需 LLM 查询翻译。

```bash
# 唯一嵌入模型
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

---

## 10. 故障排查

### 问题: MPS 嵌入时内存持续增长

**症状**: 嵌入过程中 Python 进程内存超过 35GB

**原因**: `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 默认 1.0 允许 MPS 无限制分配

**解决**:
1. 确保 `_mps_embed_subprocess.py` 已自动设置 `HIGH_WATERMARK_RATIO=0.5`
2. 确保使用 `batch_size≤16` (最优 4, 见 4.4 节 O(n²) 分析)
3. 如仍超限, 改 `EMBEDDING_DEVICE=cpu`

---

### 问题: CUDA out of memory

**症状**: `torch.cuda.OutOfMemoryError`

**解决**:
1. 减小 `--batch-size` (默认 32 → 16 或 8)
2. 检查是否有其他进程占用 GPU
3. 如 GPU 显存 < 4GB, 改用 CPU + 小型模型

---

### 问题: Windows 下 torch 未检测到 CUDA

**解决**:
1. 确认 `nvidia-smi` 可正常输出
2. 确认 PyTorch CUDA 版本: `pip list | grep torch`
3. 如显示 `torch 2.x.x+cpu`, 需重新安装 CUDA 版本:
   ```
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

### 问题: Grace CPU 上 torch 行为异常

**症状**: ImportError 或 Segmentation Fault

**可能原因**: 安装了 x86_64 版本的 PyTorch wheel

**解决**: 使用 ARM64 PyTorch 构建 (NVIDIA 官方提供 Grace 优化的 PyTorch 容器)

---

## 11. 平台检测代码 (供扩展)

```python
from src.utils.helpers import get_hardware_info, HardwareInfo

hw = get_hardware_info()
print(f"OS:       {hw.os_label}")
print(f"CPU:      {hw.cpu_label} ({hw.cpu_arch})")
print(f"GPU:      {hw.gpu_type}")
print(f"GPU Mem:  {hw.gpu_memory_mb} MB")
print(f"Device:   {hw.recommended_device}")
print(f"Unified:  {hw.is_unified_memory}")
print(f"MPS Bug:  {hw.mps_known_issues}")
```

预期输出示例:

```
OS:       macOS
CPU:      Apple Silicon (arm64)
GPU:      apple_mps
GPU Mem:  0 MB
Device:   cpu
Unified:  True
MPS Bug:  True
```

---

## 12. 配置建议速查

| 你的环境 | `.env` 配置 |
|---|---|
| NVIDIA GPU 服务器 | `EMBEDDING_DEVICE=auto`, `EMBEDDING_PROVIDER=local` |
| NVIDIA GB10 (Grace Blackwell) | `EMBEDDING_DEVICE=auto`, `EMBEDDING_PROVIDER=local`, ARM PyTorch CUDA |
| Apple Silicon Mac (批量嵌入) | `EMBEDDING_DEVICE=mps`, batch_size=4, ~1.9h (108K) |
| Apple Silicon Mac (API 服务) | `EMBEDDING_DEVICE=cpu`, `EMBEDDING_PROVIDER=api` |
| Intel/AMD 无 GPU | `EMBEDDING_DEVICE=cpu`, `EMBEDDING_PROVIDER=api` |
| NVIDIA Grace + GPU | `EMBEDDING_DEVICE=auto`, 确认 ARM PyTorch 版本 |
| Windows + NVIDIA | `EMBEDDING_DEVICE=auto`, 确认 CUDA PyTorch 版本 |

> **多语言场景**: 所有平台统一使用 BGE-M3, `LOCAL_EMBEDDING_MODEL=BAAI/bge-m3`。

> **内网离线部署**: 参见 [docs/deployment/offline-deployment.md](../deployment/offline-deployment.md)。
