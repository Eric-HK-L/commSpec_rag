#!/usr/bin/env bash
# ==================== 3GPP RAG — 离线环境一键安装 ====================
#
# 用法:
#   bash scripts/install_offline.sh                       # 自动检测平台
#   bash scripts/install_offline.sh linux-x86_64           # 指定 Intel x86
#   bash scripts/install_offline.sh linux-aarch64          # 指定 GB10 ARM
#
# 前置条件:
#   - offline/ 目录已从外网拷贝到项目根目录
#   - Python 3.11+ 已安装
#   - Docker 已安装 (向量数据库需要)
#   - 项目已 git clone
#
# 安装内容:
#   1. Python 虚拟环境 + pip 离线安装
#   2. HuggingFace 模型配置 (离线模式)
#   3. Docker 镜像导入
#   4. .env 配置生成
#   5. 验证
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OFFLINE_DIR="$PROJECT_ROOT/offline"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "=========================================================="
echo "  3GPP RAG — 离线环境安装"
echo "=========================================================="

# ── 0. 检测平台 ──
detect_platform() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64|amd64)
            echo "linux-x86_64"
            ;;
        aarch64|arm64)
            # 检测是否有 NVIDIA GPU (GB10)
            if command -v nvidia-smi &>/dev/null; then
                log "检测到 NVIDIA GPU (GB10 / Grace Blackwell)"
            else
                warn "ARM64 但未检测到 NVIDIA GPU, 将使用 CPU 模式"
            fi
            echo "linux-aarch64"
            ;;
        *)
            err "不支持的架构: $arch"
            ;;
    esac
}

TARGET_PLATFORM="${1:-$(detect_platform)}"
log "目标平台: $TARGET_PLATFORM"

# ── 0. 前置检查 ──
if [ ! -d "$OFFLINE_DIR" ]; then
    err "离线目录不存在: $OFFLINE_DIR
  请先将外网制备的 offline/ 目录拷贝到项目根目录"
fi

if [ ! -f "$OFFLINE_DIR/manifest.json" ]; then
    err "manifest.json 不存在, 离线包不完整"
fi

# ── 1. Python 虚拟环境 ──
log "创建 Python 虚拟环境..."
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    log "虚拟环境创建: $VENV_DIR"
else
    log "虚拟环境已存在, 使用现有"
fi
source "$VENV_DIR/bin/activate"

# ── 2. Pip 离线安装 ──
WHEELS_DIR="$OFFLINE_DIR/wheels/$TARGET_PLATFORM"
if [ -d "$WHEELS_DIR" ] && ls "$WHEELS_DIR"/*.whl &>/dev/null; then
    wheel_count=$(ls "$WHEELS_DIR"/*.whl | wc -l | tr -d ' ')
    log "安装 $wheel_count 个 pip 包 (离线)..."
    pip install --no-index --find-links="$WHEELS_DIR" -r "$PROJECT_ROOT/requirements.txt" 2>&1 | tail -3
    log "pip 安装完成"
else
    warn "未找到 $TARGET_PLATFORM 的 wheels, 尝试通用安装..."
    pip install -r "$PROJECT_ROOT/requirements.txt" || warn "在线安装失败, 部分功能不可用"
fi

# ── 3. HuggingFace 模型配置 ──
# 双模型支持:
#   - BGE-M3 (BAAI/bge-m3): 多语言, 1024-dim, 推荐默认
#   - bge-large-en-v1.5 (BAAI/bge-large-en-v1.5): 纯英文, 1024-dim, 英文精度略高
# 两个模型都在 offline/models/ 下时自动配置, 可在 .env 中切换 LOCAL_EMBEDDING_MODEL
MODELS_DIR="$OFFLINE_DIR/models"
if [ -d "$MODELS_DIR" ] && ls "$MODELS_DIR"/*/ &>/dev/null 2>&1; then
    log "配置 HuggingFace 离线模型路径..."

    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    mkdir -p "$HF_HOME/hub"

    for model_dir in "$MODELS_DIR"/*/; do
        model_name=$(basename "$model_dir")
        [ "$model_name" = "*" ] && continue

        hf_cache_dir="$HF_HOME/hub/models--${model_name}"
        mkdir -p "$hf_cache_dir"
        snapshots_dir="$hf_cache_dir/snapshots"
        mkdir -p "$snapshots_dir"

        snapshot_hash=$(echo "$model_name" | md5sum | cut -c1-16 2>/dev/null || echo "offline-v1")
        snapshot_path="$snapshots_dir/$snapshot_hash"

        if [ ! -L "$snapshot_path" ] && [ ! -d "$snapshot_path" ]; then
            ln -s "$model_dir" "$snapshot_path" 2>/dev/null || cp -r "$model_dir" "$snapshot_path"
        fi

        log "  模型: $model_name"
    done
else
    warn "offline/models/ 未包含模型"
    echo "  请将模型文件放到 offline/models/ 下:"
    echo "    offline/models/BAAI--bge-m3/                  (多语言, 推荐)"
    echo "    offline/models/BAAI--bge-large-en-v1.5/       (纯英文)"
fi

# ── 4. Docker 镜像导入 ──
DOCKER_DIR="$OFFLINE_DIR/docker"
if command -v docker &>/dev/null; then
    # 选择正确的架构目录
    case "$TARGET_PLATFORM" in
        linux-x86_64)  DOCKER_ARCH_DIR="$DOCKER_DIR/linux-amd64" ;;
        linux-aarch64) DOCKER_ARCH_DIR="$DOCKER_DIR/linux-arm64" ;;
        *)             DOCKER_ARCH_DIR="" ;;
    esac

    if [ -n "$DOCKER_ARCH_DIR" ] && [ -d "$DOCKER_ARCH_DIR" ]; then
        log "导入 Docker 镜像 ($DOCKER_ARCH_DIR)..."
        for tar_file in "$DOCKER_ARCH_DIR"/*.tar; do
            [ -f "$tar_file" ] || continue
            img_name=$(basename "$tar_file" .tar)
            log "  导入: $img_name"
            docker load -i "$tar_file" 2>&1 | tail -1
        done
        log "Docker 镜像导入完成"
    else
        warn "未找到 $TARGET_PLATFORM 的 Docker 镜像"
    fi
else
    warn "Docker 未安装 → 跳过镜像导入"
    echo "  安装 Docker 后手动导入:"
    echo "    docker load -i $DOCKER_DIR/linux-amd64/*.tar"
fi

# ── 5. .env 配置 ──
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    log "生成 .env 配置文件..."
    # 根据平台选择嵌入设备
    case "$TARGET_PLATFORM" in
        linux-x86_64)
            EMBED_DEVICE="cpu"
            ;;
        linux-aarch64)
            EMBED_DEVICE="auto"  # GB10 有 CUDA
            ;;
        *)
            EMBED_DEVICE="cpu"
            ;;
    esac

    cat > "$PROJECT_ROOT/.env" << ENVEOF
# 3GPP RAG — 离线环境配置
# 由 install_offline.sh 自动生成 ($(date))

# LLM (填写公司内网 API 地址)
LLM_BASE_URL=https://your-company-llm-api/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model-name

# ── 嵌入模型 (双模型支持, 离线本地 BGE) ──
# 嵌入后端: "local"=本地BGE模型 / "api"=云端API
EMBEDDING_PROVIDER=local
# 设备: "cpu"(通用) / "auto"(自动,GB10选CUDA) / "mps"(macOS,仅本地开发)
EMBEDDING_DEVICE=$EMBED_DEVICE
# 本地模型选择 (两者均为 1024-dim, 根据场景切换):
#   BAAI/bge-m3              多语言 (100+语言), 推荐默认
#   BAAI/bge-large-en-v1.5   纯英文, 英文精度略高
# 注意: macOS MPS 仅支持单进程 GPU (代码层已硬编码 workers=1)
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3

# 离线模式 (禁止网络请求)
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1

# 向量数据库
VECTOR_DB=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=TeleComm_specs

# 文档摄入
INGESTION_SOURCE=from_scratch
DOCUMENTS_DIR=data/documents/R18
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# 检索
MAX_SEARCH_RESULTS=10
DENSE_TOP_K=100
BM25_TOP_K=100
SIMILARITY_THRESHOLD=0.7

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# 日志
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
ENVEOF

    log ".env 已生成, 请填入公司 LLM API 信息: LLM_BASE_URL / LLM_API_KEY / LLM_MODEL"
else
    log ".env 已存在, 跳过"
fi

# ── 6. 验证 ──
echo ""
echo "=========================================================="
echo "  验证安装"
echo "=========================================================="

# Python 包
python -c "import fastapi, pymilvus, sentence_transformers, torch; print('  ✅ Python 核心包')" 2>&1 || warn "Python 包导入失败"
python -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')" 2>&1 || true

# 模型 (公司内部提供, 可选)
MODELS_DIR="$OFFLINE_DIR/models"
for model in "BAAI--bge-large-en-v1.5" "BAAI--bge-m3"; do
    model_dir="$MODELS_DIR/$model"
    if [ -d "$model_dir" ] && ls "$model_dir"/*.safetensors &>/dev/null 2>&1; then
        log "模型: $model"
    fi
done
if [ ! -d "$MODELS_DIR" ] || ! ls "$MODELS_DIR"/*/*.safetensors &>/dev/null 2>&1; then
    warn "模型未包含在离线包中 (预期: BGE-M3 或 bge-large-en-v1.5)"
    echo "  请将模型文件放到 offline/models/ 下 (参见上方说明)"
fi

# Docker
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    log "Docker: 运行中"
else
    warn "Docker: 未运行 (向量数据库需要)"
fi

echo ""
echo "=========================================================="
echo "  安装完成!"
echo "=========================================================="
echo ""
echo "  下一步:"
echo "    1. 编辑 .env, 填入公司 LLM API 信息"
echo "    2. 启动 Milvus:  docker compose -f deploy/docker-compose.yml up -d etcd minio milvus"
echo "    3. 摄入文档:    python scripts/bulk_ingest.py --full-rebuild"
echo "    4. 启动服务:    python -m src.main"
echo "    5. 访问前端:    http://localhost:3000"
echo ""
