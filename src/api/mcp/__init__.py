"""
MCP（Model Context Protocol）工具集

将核心 RAG 能力封装为 MCP 工具，供 AI Agent 直接调用：
  - search_specifications          — 规范全文搜索
  - get_specification_details      — 规范详情与元数据
  - compare_specifications         — 多版本对比
  - find_implementation_requirements — 实现需求提取

参考来源：
- 3gpp-mcp-server-main/src/tools/search-specifications.ts
- 3gpp-mcp-server-main/src/tools/get-specification-details.ts
- 3gpp-mcp-server-main/src/tools/compare-specifications.ts
- 3gpp-mcp-server-main/src/tools/find-implementation-requirements.ts
"""
