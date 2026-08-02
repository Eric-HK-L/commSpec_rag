# 3GPP RAG — 内网离线部署手册

> **适用场景**：公司内网可访问 GitHub，但无法访问 PyPI / HuggingFace / Docker Hub。
> **目标读者**：负责将本系统部署到公司内网服务器的工程师。
> **最后更新**：2026-07-14

---

## 目录

- [0. 部署架构总览](#0-部署架构总览)
- [1. 前置准备清单](#1-前置准备清单)
- [2. 第一章：外网制备离线包](#2-第一章外网制备离线包)
- [3. 第二章：离线包传输到内网](#3-第二章离线包传输到内网)
- [4. 第三章：内网环境安装](#4-第三章内网环境安装)
- [5. 第四章：BGE-M3 模型配置](#5-第四章bge-m3-模型配置)
- [6. 第五章：启动服务与验收](#6-第五章启动服务与验收)
- [7. 第六章：日常运维](#7-第六章日常运维)
- [附录 A：故障排查](#附录-a故障排查)
- [附录 B：快速参考卡](#附录-b快速参考卡)
- [附录 C：离线包目录结构](#附录-c离线包目录结构)

---

## 0. 部署架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                      外网机器 (可访问互联网)                        │
│                                                                  │
│  ① git clone <项目仓库>                                           │
│  ② python scripts/prepare_offline.py                              │
│     ├── 下载 pip wheels (Intel x86_64 + NVIDIA GB10 ARM64)        │
│     ├── 导出 Docker 镜像 (Milvus + MinIO + etcd)                  │
│     └── 生成 manifest.json 清单                                   │
│                                                                  │
│  ③ 产物: offline/ 目录  (~3 GB, 不含模型)                         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                    USB / 网盘 / 内网共享
                           │
┌──────────────────────────┴───────────────────────────────────────┐
│                      内网机器 (公司服务器)                          │
│                                                                  │
│  ④ git clone <项目仓库>                                           │
│  ⑤ 拷贝 offline/ 到项目根目录                                     │
│  ⑥ bash scripts/install_offline.sh                                │
│     ├── 创建 Python .venv                                         │
│     ├── 离线安装 pip 包                                           │
│     ├── 导入 Docker 镜像                                           │
│     └── 生成 .env 配置                                             │
│                                                                  │
│  ⑦ 放入公司 BGE-M3 模型 → offline/models/                         │
│  ⑧ docker compose up -d etcd minio milvus                        │
│  ⑨ python scripts/bulk_ingest.py --full-rebuild                  │
│  ⑩ python -m src.main                                             │
│                                                                  │
│  ✅ 访问 http://<服务器IP>:8000/docs                               │
└──────────────────────────────────────────────────────────────────┘
```

### 目标平台

| 平台标识 | 硬件 | 嵌入设备 | Docker 架构 |
|---|---|---|---|
| `linux-x86_64` | Intel/AMD x64 CPU, 无 GPU | `cpu` | `linux/amd64` |
| `linux-aarch64` | NVIDIA GB10 (ARM64 + Blackwell GPU) | `auto` → `cuda` | `linux/arm64` |

> **BGE-M3 模型不包含在离线包中** — 由公司内部提供。若公司有 HuggingFace 镜像或内部模型仓库，安装时配置 `HF_HOME` 指向即可。

---

## 1. 前置准备清单

在开始部署前，确认以下条件：

### 外网机器

- [ ] 可访问 GitHub（clone 项目）
- [ ] 可访问 PyPI（下载 pip wheels）
- [ ] 可访问 Docker Hub（pull 镜像）
- [ ] Python 3.11+ 已安装
- [ ] Docker 24+ 已安装并运行
- [ ] 磁盘空间 ≥ 10 GB（离线包 ~3 GB + 临时文件）

### 内网机器

- [ ] 可访问 GitHub（clone 项目）
- [ ] Python 3.11+ 已安装
- [ ] Docker 24+ 已安装
- [ ] 磁盘空间 ≥ 20 GB（模型 2.2 GB + Milvus 数据 + 文档）
- [ ] 内存 ≥ 16 GB（推荐 20 GB+，含 BGE-M3）
- [ ] 公司 LLM API 端点已知（URL / Key / Model 名）

### 传输介质

- [ ] U 盘 / 移动硬盘 ≥ 10 GB（或内网共享目录可达）
- [ ] 若走网络传输：确认两台机器网络互通

---

## 2. 第一章：外网制备离线包

在**有完整互联网访问**的机器上执行。

### 2.1 克隆项目

```bash
git clone <项目仓库地址>
cd 3GPP_RAG_project
```

### 2.2 安装制备工具依赖

```bash
# 制备脚本仅需标准库，无需额外 pip 包
# 若需下载 HuggingFace 模型（公司有则跳过）：
pip install huggingface_hub
```

### 2.3 执行制备

```bash
# 方案 A：全量制备（Intel + GB10 双平台，推荐）
python scripts/prepare_offline.py

# 方案 B：仅制备 Intel 平台
python scripts/prepare_offline.py --platform linux-x86_64

# 方案 C：仅制备 GB10 平台
python scripts/prepare_offline.py --platform linux-aarch64

# 按需组合
python scripts/prepare_offline.py --pip-only       # 仅 pip wheels
python scripts/prepare_offline.py --docker-only    # 仅 Docker 镜像
python scripts/prepare_offline.py --models-only    # 仅 HuggingFace 模型
```

**执行时间预估**：

| 操作 | 耗时 | 说明 |
|---|---|---|
| pip wheels (单平台) | 5-15 分钟 | 取决于网速，~50 个包 |
| pip wheels (双平台) | 10-25 分钟 | Intel + ARM64 并行 |
| Docker 镜像 (单架构) | 5-15 分钟 | ~2 GB 下载 |
| Docker 镜像 (双架构) | 8-20 分钟 | amd64 + arm64 |
| HuggingFace 模型 | 5-15 分钟 | bge-m3 ~2.2 GB |
| **总计（不含模型）** | **~15-40 分钟** | |

### 2.4 预期输出

```
============================================================
  离线依赖包制备完成
============================================================
  📦 linux-x86_64 (Intel x86_64 Linux (无 GPU)): 48 个 wheel, 780 MB
  📦 linux-aarch64 (NVIDIA GB10 (ARM64 + CUDA)): 50 个 wheel, 920 MB
  🐳 Docker linux-amd64: 1800 MB
  🐳 Docker linux-arm64: 1950 MB

  📁 总大小: ~5450 MB (5.3 GB)
  📂 目录: /path/to/3GPP_RAG_project/offline

  下一步:
    1. 将 offline/ 目录拷贝到内网机器
    2. Intel x86: bash scripts/install_offline.sh linux-x86_64
    3. GB10 ARM:  bash scripts/install_offline.sh linux-aarch64
```

### 2.5 验证离线包完整性

```bash
# 检查 manifest 是否存在
cat offline/manifest.json | python -m json.tool | head -20

# 统计 wheels 数量
echo "Intel:  $(ls offline/wheels/linux-x86_64/*.whl 2>/dev/null | wc -l) 个"
echo "GB10:   $(ls offline/wheels/linux-aarch64/*.whl 2>/dev/null | wc -l) 个"

# 统计 Docker 镜像
echo "amd64:  $(ls offline/docker/linux-amd64/*.tar 2>/dev/null | wc -l) 个"
echo "arm64:  $(ls offline/docker/linux-arm64/*.tar 2>/dev/null | wc -l) 个"

# 检查总大小
du -sh offline/
```

---

## 3. 第二章：离线包传输到内网

### 3.1 传输 offline/ 目录

```
方式 A — U 盘 / 移动硬盘:
  1. 将 offline/ 目录完整拷贝到移动介质
  2. 在内网机器上拷贝到项目根目录

方式 B — 内网共享 (如 Samba / NFS):
  1. 将 offline/ 上传到内网共享目录
  2. 在内网机器上下载到项目根目录

方式 C — scp (若两台机器网络互通):
  scp -r offline/ user@internal-server:/path/to/3GPP_RAG_project/
```

### 3.2 内网克隆项目

```bash
# 内网机器上
git clone <项目仓库地址>
cd 3GPP_RAG_project
```

### 3.3 放置离线包

```bash
# 将 offline/ 目录放到项目根目录
# 最终结构:
#   3GPP_RAG_project/
#   ├── offline/          ← 从外网拷贝来的
#   ├── scripts/
#   ├── src/
#   └── ...
```

---

## 4. 第三章：内网环境安装

### 4.1 确认环境

```bash
# 确认 Python 版本 (需 ≥ 3.11)
python3 --version

# 确认 Docker
docker --version
docker info | grep "Architecture"

# 确认离线包存在
ls offline/manifest.json && echo "✅ 离线包就绪" || echo "❌ 缺少离线包"
```

### 4.2 执行安装

```bash
# 自动检测平台并安装
bash scripts/install_offline.sh

# 或显式指定平台
bash scripts/install_offline.sh linux-x86_64     # Intel 服务器
bash scripts/install_offline.sh linux-aarch64    # GB10 服务器
```

### 4.3 安装过程详解

脚本按以下顺序自动执行，每步均有绿色 `[OK]` 或黄色 `[WARN]` 提示：

| 步骤 | 操作 | 预期输出 |
|---|---|---|
| 0 | 平台检测 | `[OK] 目标平台: linux-x86_64` |
| 1 | 创建 `.venv` | `[OK] 虚拟环境创建: .../.venv` |
| 2 | pip 离线安装 | `[OK] 安装 48 个 pip 包 (离线)...` |
| 3 | 模型配置 | `[WARN] 模型未包含（公司内部提供）` |
| 4 | Docker 导入 | `[OK] Docker 镜像导入完成` |
| 5 | `.env` 生成 | `[OK] .env 已生成` |
| 6 | 验证 | `✅ numpy x.x.x` / `✅ Python 核心包` |

### 4.4 安装后验证

```bash
# 激活虚拟环境
source .venv/bin/activate

# 验证核心包
python -c "
import fastapi, pymilvus, sentence_transformers, torch
print('✅ 核心包导入成功')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
"

# 验证 Docker 镜像
docker images | grep -E "milvus|minio|etcd"
```

**预期输出**（Intel 平台）：
```
✅ 核心包导入成功
  PyTorch: 2.5.1
  CUDA 可用: False

REPOSITORY          TAG                  IMAGE ID       SIZE
milvusdb/milvus     v2.4.10             abc123def456   1.48GB
minio/minio         RELEASE.2023-03...   def456abc789   440MB
quay.io/coreos/etcd v3.5.5              789def456abc   180MB
```

**预期输出**（GB10 平台）：
```
✅ 核心包导入成功
  PyTorch: 2.5.1+cu128
  CUDA 可用: True
```

---

## 5. 第四章：BGE-M3 模型配置

### 5.1 前提

BGE-M3 嵌入模型由公司内部提供，不包含在离线包中。请从公司内部获取完整的模型文件。

### 5.2 方案 A：放入离线模型目录（推荐）

```bash
# 将公司提供的 BGE-M3 模型文件放到:
offline/models/BAAI--bge-m3/

# 目录结构:
offline/models/BAAI--bge-m3/
├── config.json
├── config_sentence_transformers.json
├── model.safetensors          # 或 pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── sentence_bert_config.json
├── modules.json
├── 1_Pooling/
│   └── config.json
└── ...
```

然后重新运行安装脚本，它会自动链接到 HuggingFace 缓存：

```bash
bash scripts/install_offline.sh
```

### 5.3 方案 B：使用公司内部模型仓库

若公司已有 HuggingFace 镜像或内部模型仓库：

```bash
# 设置指向公司内部仓库
export HF_HOME=/path/to/company/huggingface/mirror
export HF_ENDPOINT=https://hf-mirror.internal.company.com

# 或写入 .env
echo "HF_HOME=/path/to/company/huggingface/mirror" >> .env
echo "TRANSFORMERS_OFFLINE=1" >> .env
echo "HF_HUB_OFFLINE=1" >> .env
```

### 5.4 验证模型可用

```bash
source .venv/bin/activate

python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3', local_files_only=True)
print(f'✅ BGE-M3 加载成功')
print(f'  维度: {model.get_sentence_embedding_dimension()}')
# 输出: 维度: 1024
"
```

---

## 6. 第五章：启动服务与验收

### 6.1 编辑 .env 配置

```bash
vim .env
```

必须填写的三项：

```ini
LLM_BASE_URL=https://llm.internal.company.com/v1   # 公司 LLM API 地址
LLM_API_KEY=your-company-api-key                     # API Key
LLM_MODEL=company-model-name                         # 模型名
```

> **数据目录说明：** 所有运行时数据（向量、清单、检查点、嵌入缓存）统一存放在 `DATA_DIR` 下（默认 `./data/`）。
> 如需外置存储，取消 `.env` 中 `DATA_DIR` 注释并改为绝对路径（如 `/mnt/nas/3gpp-rag-data`）。
> 摄入过程会产生以下子目录：
> - `data/vectors/` — BM25 索引
> - `data/manifest/` — 摄入清单（SHA256 + 版本号）
> - `data/checkpoint/` — 断点续传进度
> - `data/cache/` — 嵌入缓存（`embedding_cache.db`）

### 6.2 启动向量数据库 (Milvus)

```bash
# 创建 Docker volumes
docker volume create spec_rag_etcd_data
docker volume create spec_rag_minio_data
docker volume create spec_rag_milvus_data

# 启动 etcd + MinIO + Milvus
docker compose -f deploy/docker-compose.yml up -d etcd minio milvus

# 等待 Milvus 健康检查通过 (约 90 秒)
docker compose -f deploy/docker-compose.yml ps
```

**预期输出**：
```
NAME                     STATUS                    PORTS
spec_rag-etcd-1            Up (healthy)              ...
spec_rag-minio-1           Up (healthy)              0.0.0.0:9000-9001->9000-9001/tcp
spec_rag-milvus-1          Up (healthy)              0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
```

### 6.3 验证 Milvus 连接

```bash
source .venv/bin/activate

python -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='http://localhost:19530')
collections = client.list_collections()
print(f'✅ Milvus 连接成功')
print(f'  Collections: {collections}')
"
```

### 6.4 摄入 3GPP 文档

```bash
# 全量重建（首次部署）
python scripts/bulk_ingest.py --full-rebuild

# 增量摄入（日常新增/修改文档时使用，默认模式）
python scripts/bulk_ingest.py

# 断点续传（摄入中断后恢复，跳过已完成文件与已嵌入 chunk）
python scripts/bulk_ingest.py --resume-from-checkpoint
```

**执行时间预估**：

| 平台 | 模型 | 预估耗时 |
|---|---|---|
| Intel x64 CPU | BGE-M3 | ~45-70 分钟（取决于 CPU 核数） |
| NVIDIA GB10 | BGE-M3 (CUDA) | ~3-5 分钟 |
| NVIDIA A100 | BGE-M3 (CUDA) | ~2-3 分钟 |

> 进度条会实时显示 `specs X/Y | chunks A/B`，完成后输出摘要统计。
> 摄入过程中会在 `data/checkpoint/` 保存进度，中断后可用 `--resume-from-checkpoint` 恢复。

### 6.5 启动 API 服务

```bash
# 前台运行（调试）
python -m src.main

# 后台运行（生产）
nohup python -m src.main > logs/api.log 2>&1 &
```

### 6.6 验收测试

```bash
# 1. API 健康检查
curl http://localhost:8000/api/v1/health

# 预期输出:
# {"status":"ready","vector_db":"MilvusStore","chunk_count":108000}

# 2. 检索测试
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the 5G NR physical layer structure?", "top_k": 5}'

# 预期输出: JSON 数组，含 document_id / content / score 字段

# 3. 运行评估（可选）
python tests/eval/run_eval.py
```

### 6.7 （可选）启动前端

```bash
cd frontend
npm install
npm run build
npm start
# 访问: http://localhost:3000
```

---

## 7. 第六章：日常运维

### 7.1 更新依赖

当 `requirements.txt` 有变更时：

```bash
# === 外网机器 ===
git pull
python scripts/prepare_offline.py --pip-only
# 拷贝 offline/wheels/ 到内网

# === 内网机器 ===
git pull
bash scripts/install_offline.sh
```

### 7.2 增量摄入新文档

```bash
# 放入新 .docx 文件到 data/documents/R18/<series>/
# 默认模式即为增量：仅处理新增/修改的文档
python scripts/bulk_ingest.py
```

### 7.3 重启服务

```bash
# 停止
docker compose -f deploy/docker-compose.yml down

# 启动
docker compose -f deploy/docker-compose.yml up -d etcd minio milvus
python -m src.main &
```

### 7.4 日志查看

```bash
# API 日志
tail -f logs/app.log

# Milvus 日志
docker logs -f spec_rag-milvus-1

# Docker 状态
docker compose -f deploy/docker-compose.yml ps
```

---

## 附录 A：故障排查

### A.1 pip 安装报 "not a supported wheel"

```
ERROR: torch-2.5.1-cp311-cp311-linux_x86_64.whl is not a supported wheel
```

**原因**：目标平台与 wheel 平台不匹配。\
**解决**：在外网重新制备，用 `--platform` 指定正确平台：

```bash
python scripts/prepare_offline.py --platform linux-x86_64 --pip-only
```

### A.2 Docker 无法启动 Milvus — 架构不匹配

```
WARNING: The requested image's platform (linux/amd64) does not match
the detected host platform (linux/arm64)
```

**解决**：确认使用正确架构的镜像。

```bash
# 查看当前平台
uname -m                    # x86_64 或 aarch64

# 加载正确架构的镜像
docker load -i offline/docker/linux-amd64/*.tar    # Intel
docker load -i offline/docker/linux-arm64/*.tar    # GB10
```

### A.3 Milvus 健康检查失败

```
spec_rag-milvus-1  Up (health: starting) 或 unhealthy
```

**排查步骤**：

```bash
# 1. 检查 etcd + minio 是否都 healthy
docker compose -f deploy/docker-compose.yml ps

# 2. 查看 Milvus 日志
docker logs spec_rag-milvus-1 --tail 50

# 3. 如果持续 unhealthy, 重建
docker compose -f deploy/docker-compose.yml down -v
docker volume rm spec_rag_etcd_data spec_rag_minio_data spec_rag_milvus_data
docker volume create spec_rag_etcd_data
docker volume create spec_rag_minio_data
docker volume create spec_rag_milvus_data
docker compose -f deploy/docker-compose.yml up -d etcd minio milvus
```

### A.4 模型加载 "We couldn't connect to HuggingFace"

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file
```

**原因**：`.env` 中未设置离线模式环境变量。\
**解决**：

```bash
# 确认 .env 包含:
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

或在代码中显式指定：

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
```

### A.5 GB10 上 PyTorch CUDA 不可用

```
>>> torch.cuda.is_available()
False
```

**原因**：GB10 需要 ARM64 版本的 PyTorch CUDA wheel。\
**解决**：在外网制备时用 `--platform linux-aarch64`，脚本会自动从 NVIDIA PyTorch 索引下载 `torch==2.5.1+cu128` 的 ARM wheel。

### A.6 内存不足 (OOM)

```
RuntimeError: [enforce fail at ..] fail, likely out of memory
```

**解决**：
1. 减小分块大小：编辑 `.env` → `CHUNK_SIZE=256`（降低单次嵌入内存压力）
2. 确认内存 ≥ 16 GB（BGE-M3 推荐 20 GB+）

### A.7 离线包不完整 / 缺少包

```bash
# 检查 manifest
cat offline/manifest.json | python -c "
import json, sys
m = json.load(sys.stdin)
for plat, info in m['target_platforms'].items():
    print(f'{plat}: {info[\"wheel_count\"]} wheels, {info[\"size_mb\"]} MB')
for arch, info in m.get('docker', {}).items():
    print(f'Docker {arch}: {info[\"images\"]}, {info[\"size_mb\"]} MB')
"
```

### A.8 摄入中断后恢复

```
摄入过程意外中断（如 OOM、断电）
```

**解决**：使用断点续传从 `data/checkpoint/` 自动恢复。

```bash
# 续传：跳过已完成的文档提取与已嵌入的 chunks
python scripts/bulk_ingest.py --resume-from-checkpoint
```

> 检查点记录在 `data/checkpoint/` 下，包含已处理文件列表与 Milvus 已入库 chunk 数。

---

## 附录 B：快速参考卡

### 外网制备

```bash
git clone <repo> && cd 3GPP_RAG_project
python scripts/prepare_offline.py                     # 双平台全量
python scripts/prepare_offline.py --platform linux-x86_64  # 仅 Intel
```

### 内网安装

```bash
git clone <repo> && cd 3GPP_RAG_project
# 放入 offline/ 目录
bash scripts/install_offline.sh                       # 自动检测平台
```

### 启动服务

```bash
docker compose -f deploy/docker-compose.yml up -d etcd minio milvus
python scripts/bulk_ingest.py --full-rebuild
python -m src.main
```

### 验收

```bash
curl http://localhost:8000/api/v1/health              # → {"status":"ready",...}
curl -X POST http://localhost:8000/api/v1/search \    # → JSON 结果
  -H "Content-Type: application/json" \
  -d '{"query": "5G NR physical layer", "top_k": 5}'
```

### 平台速查

| 服务器 | 安装命令 | 嵌入设备 |
|---|---|---|
| Intel CPU | `bash scripts/install_offline.sh linux-x86_64` | `cpu` |
| NVIDIA GB10 | `bash scripts/install_offline.sh linux-aarch64` | `auto` → `cuda` |

---

## 附录 C：离线包目录结构

```
offline/
├── manifest.json                      ← 清单（含各组件大小、文件数）
├── wheels/
│   ├── linux-x86_64/                  ← Intel 服务器 pip 包
│   │   ├── torch-2.5.1-cp311-...whl
│   │   ├── fastapi-0.115.6-...whl
│   │   └── ...（~50 个 .whl）
│   └── linux-aarch64/                 ← GB10 pip 包（含 CUDA torch）
│       ├── torch-2.5.1+cu128-...whl
│       └── ...（~52 个 .whl）
├── models/                            ← 公司内部提供（可选）
│   └── BAAI--bge-m3/
│       ├── model.safetensors
│       ├── config.json
│       └── ...
└── docker/
    ├── linux-amd64/                   ← Intel Docker 镜像
    │   ├── milvus-v2.4.10.tar
    │   ├── minio-20230320.tar
    │   └── etcd-v3.5.5.tar
    └── linux-arm64/                   ← GB10 Docker 镜像
        ├── milvus-v2.4.10.tar
        ├── minio-20230320.tar
        └── etcd-v3.5.5.tar
```

---

> **相关文档**：[硬件兼容性指南](../design/hardware-compatibility.md) | [架构设计](../design/architecture.md)
