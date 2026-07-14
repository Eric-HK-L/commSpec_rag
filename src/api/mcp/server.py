"""MCP Server — JSON-RPC 2.0 stdio 传输, 兼容 Claude Desktop.

用法:
    python -m src.api.mcp.server          # stdio 模式 (Claude Desktop)
    python -m src.api.mcp.server --sse    # SSE 模式 (远程 Agent, 端口 8001)

Claude Desktop 配置 (~/Library/Application Support/Claude/claude_desktop_config.json):
{
  "mcpServers": {
    "3gpp-rag": {
      "command": "/path/to/.venv/bin/python3",
      "args": ["-m", "src.api.mcp.server"],
      "cwd": "/path/to/3GPP_RAG_project"
    }
  }
}
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcp")

# 添加项目根目录到 path (确保直接运行也能导入)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ── JSON-RPC 消息处理 ──

class MCPServer:
    """最小 MCP Server 实现 — 支持 tools/list 和 tools/call."""

    def __init__(self):
        from src.api.mcp.tools import TOOL_SCHEMAS, MCPToolHandler, set_global_store
        self._tool_schemas = TOOL_SCHEMAS
        self._handler = MCPToolHandler()
        self._set_store = set_global_store
        self._initialized = False

    def init_store(self) -> None:
        """延迟初始化 Milvus 向量库连接."""
        if self._initialized:
            return
        from src.config import settings
        from src.retriever.milvus_store import MilvusStore
        store = MilvusStore(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection_name,
        )
        store.connect()
        self._set_store(store)
        self._initialized = True
        logger.info("MCP 向量库已连接: %s (%d chunks)", store.__class__.__name__, store.count)

    def handle_message(self, message: dict) -> dict | None:
        """处理单条 JSON-RPC 消息, 返回响应或 None (通知)."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        # 通知 (无 id) → 不响应
        if msg_id is None:
            self._handle_notification(method, params)
            return None

        try:
            result = self._dispatch(method, params)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }
        except Exception as e:
            logger.error("MCP 工具调用失败 (%s): %s", method, e)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": str(e)},
            }

    def _dispatch(self, method: str, params: dict) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "3gpp-rag-mcp",
                    "version": "0.2.0",
                },
            }
        elif method == "tools/list":
            return {"tools": self._tool_schemas}
        elif method == "tools/call":
            self.init_store()
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            content = self._handler.handle_tool_call(name, arguments)
            return {"content": content}
        elif method == "notifications/initialized":
            return {}
        else:
            raise ValueError(f"未知方法: {method}")

    def _handle_notification(self, method: str, params: dict) -> None:
        if method == "notifications/initialized":
            logger.info("MCP 客户端已连接")


# ── stdio 传输 ──

def run_stdio():
    """标准输入/输出 JSON-RPC 循环."""
    logging.basicConfig(
        level=logging.WARNING,  # MCP stdio 不可向 stdout 打印日志
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    server = MCPServer()
    logger.info("3GPP RAG MCP Server 启动 (stdio)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error("无效 JSON: %s", e)
            continue

        response = server.handle_message(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


# ── 入口 ──

if __name__ == "__main__":
    if "--sse" in sys.argv:
        print("SSE 模式暂未实现, 使用 stdio 模式")
    run_stdio()
