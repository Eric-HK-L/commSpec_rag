from .settings import ingestion_config, settings

__all__ = ["settings", "ingestion_config"]
"""
配置管理模块

融合 gpp-RAG-app 的 .env 加载模式与 Chat3GPP 的类配置风格，
统一管理 LLM API Key、向量数据库连接、分块参数、模型选择等配置项。

参考来源：
- gpp-RAG-app/server-main/server-app/config/settings.py  — python-dotenv + Config 类模式
- gpp-RAG-app/app-main/config.py                         — 环境变量校验与默认值
- Chat3GPP-master/configs/model_configs.py               — 纯 Python 字典集中定义
- specpilot-main/.env                                    — 最完整的 .env 模板示例
"""
