#!/usr/bin/env bash
# ── Commspec RAG 全栈启动脚本 ──
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

echo "=== Commspec RAG 全栈启动 ==="
echo ""

# 1. 检查 Docker
if ! command -v docker &>/dev/null; then
  echo "❌ 需要安装 Docker Desktop"
  exit 1
fi

# 2. 创建外部卷（如果不存在）
echo "📦 检查数据卷..."
for vol in spec_rag_etcd_data spec_rag_minio_data spec_rag_milvus_data; do
  if ! docker volume inspect "$vol" &>/dev/null; then
    echo "  创建卷: $vol"
    docker volume create "$vol"
  else
    echo "  ✓ $vol"
  fi
done

# 3. 启动基础设施
echo ""
echo "🚀 启动 Milvus 基础设施 (etcd + minio + milvus)..."
docker compose up -d etcd minio milvus

# 4. 等待 Milvus 健康
echo "⏳ 等待 Milvus 就绪..."
for i in $(seq 1 30); do
  if docker compose exec -T milvus curl -sf http://localhost:9091/healthz &>/dev/null; then
    echo "✅ Milvus 已就绪"
    break
  fi
  sleep 2
done

# 5. 启动 API + 前端
echo ""
echo "🚀 启动 API 服务 + 前端..."
docker compose up -d api frontend

# 6. 等待 API 健康
echo "⏳ 等待 API 就绪..."
for i in $(seq 1 15); do
  if curl -sf http://localhost:8000/api/v1/health &>/dev/null; then
    echo "✅ API 已就绪 (http://localhost:8000)"
    break
  fi
  sleep 2
done

# 7. 等待前端就绪
echo "⏳ 等待前端就绪..."
for i in $(seq 1 15); do
  if curl -sf http://localhost:3000 &>/dev/null; then
    echo "✅ 前端已就绪 (http://localhost:3000)"
    break
  fi
  sleep 2
done

echo ""
echo "=== 全部服务已启动 ==="
echo "  📡 API:      http://localhost:8000"
echo "  🖥️  前端:     http://localhost:3000"
echo "  📋 Swagger:  http://localhost:8000/docs"
echo "  ⚙️  管理后台: http://localhost:3000/admin"
