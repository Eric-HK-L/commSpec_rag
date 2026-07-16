#!/usr/bin/env bash
# ── 3GPP RAG 全栈停止脚本 ──
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

echo "=== 停止 3GPP RAG 全栈服务 ==="
docker compose down --remove-orphans
echo "✅ 全部服务已停止"
echo ""
echo "💡 提示: 数据卷保留，下次启动可复用。如需清除数据:"
echo "   docker volume rm spec_rag_etcd_data spec_rag_minio_data spec_rag_milvus_data"
