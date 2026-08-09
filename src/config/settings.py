"""应用配置 — 基于 pydantic-settings 的环境变量管理."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

# ── MPS 内存水位线 ── (必须在 torch 加载前设置)
# 默认 HIGH=1.0 允许 Metal 使用全部系统内存, 设为 0.5 上限 50%
# 已验证: batch_size=4 + HIGH=0.5 + LOW=0.3 → wired 仅 8GB
if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
if not os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"):
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.3"

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，自动从 .env 文件加载."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 项目路径 ──
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    # 数据目录 — 所有向量/文档/清单的根目录, 可配置为外部存储路径
    data_dir: str = "data"

    # ── LLM 配置 ──
    llm_base_url: str = "https://api.openai.com/v1"
    # API key: 优先 .env 的 LLM_API_KEY; 为空/占位符时自动从 llm_api_key_file 读取,
    # 避免把真实密钥写进 .env 或提交到版本库
    llm_api_key: str = ""
    # 密钥文件路径 (支持 ~ 展开), 默认读取本机 DeepSeek 密钥文件
    llm_api_key_file: str = "~/ds-api-key"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0
    llm_max_retries: int = 2  # OpenAI 客户端对 429/5xx 的自动重试次数

    @model_validator(mode="after")
    def _resolve_llm_api_key(self) -> "Settings":
        """API key 回退: .env 为空或占位符时从 llm_api_key_file 读取."""
        if self.llm_api_key and self.llm_api_key != "sk-your-key-here":
            return self
        key_path = Path(self.llm_api_key_file).expanduser()
        if key_path.is_file():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                self.llm_api_key = key
        return self

    # ── 嵌入模型 ──
    embedding_provider: Literal["api", "local"] = "local"  # 默认本地 BGE-M3
    embedding_model: str = "text-embedding-3-small"  # API 模式使用的云端模型
    embedding_dimension: int = 1024  # BGE-M3 输出维度
    embedding_device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu" — auto 自动选最优
    # 本地嵌入模型 — BGE-M3 (多语言, 1024-dim, 稠密+稀疏双向量)
    local_embedding_model: str = "BAAI/bge-m3"

    # ── Cross-Encoder Reranker ──
    reranker_enabled: bool = True  # 是否启用第二阶段 Cross-Encoder 精排
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # 本地 reranker 模型路径或 HuggingFace ID
    reranker_top_k: int = 100  # 送入 reranker 的候选数 (混合检索结果取 top-N)
    reranker_device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"

    @property
    def resolved_embedding_device(self) -> str:
        """解析 embedding_device: 'auto' → 运行时自动检测, 否则直接用配置值.

        自动检测策略 (get_best_device):
        - NVIDIA CUDA → "cuda"
        - Apple MPS   → "cpu" (安全默认, 因 MPS 内存泄漏风险)
        - 其他         → "cpu"
        """
        if self.embedding_device == "auto":
            from src.utils.helpers import get_best_device, get_hardware_info
            hw = get_hardware_info()
            device = get_best_device()
            if hw.mps_known_issues:
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "embedding_device='auto' 在 Apple Silicon 上自动降级为 'cpu'。"
                    "原因: MPS empty_cache() 只释放 Metal 命令缓冲区, 不保证立即回收统一内存,"
                    "大批量嵌入(>1000 chunks)仍会导致内存累积 (实测 57GB+ 超限, 35GB 上限)。"
                    "如仅处理小批量 (<1000 chunks), 可设 EMBEDDING_DEVICE=mps 强制启用。"
                )
            return device
        if self.embedding_device == "mps":
            from src.utils.helpers import get_hardware_info
            hw = get_hardware_info()
            if hw.mps_known_issues:
                logger = __import__("logging").getLogger(__name__)
                logger.warning(
                    "embedding_device='mps' 已显式启用。"
                    "注意: MPS empty_cache() 释放不彻底, 大批量嵌入可能内存超限。"
                    "监控内存使用, 如有问题改回 cpu。"
                )
        return self.embedding_device

    # ── 文档摄入 ──
    # 默认 from_scratch: 从本地 DOCX 直接嵌入 (HuggingFace precomputed 已废弃)
    ingestion_source: Literal["from_scratch"] = "from_scratch"
    # 以下为历史遗留 (precomputed/hybrid 模式已废弃, .env 中不再需要配置)
    pre_computed_path: str = "data/3GPP-R18"
    pre_computed_series: str = ""

    # ── 向量数据库 ──
    vector_db: Literal["milvus"] = "milvus"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "TeleComm_specs"

    # ── 文档处理 ──
    # chunk_size/chunk_overlap 已迁移至 IngestionConfig (摄入管线专用配置)
    documents_dir: str = "data/documents"

    # ── 检索 ──
    max_search_results: int = 20  # 对比类问题需要更多候选覆盖多规范
    dense_top_k: int = 100
    bm25_top_k: int = 100
    similarity_threshold: float = 0.7
    enable_nn_router: bool = False
    enable_online_search: bool = False
    # 在线搜索补充配置
    google_api_key: str = ""
    google_cse_id: str = ""
    tspec_llm_url: str = ""  # TSpec-LLM API 端点
    online_score_threshold: float = 0.6  # 离线分低于此值触发在线补充

    # ── Release 监控 ──
    release_monitor_interval_minutes: int = 120  # 文档变更检测间隔 (0=禁用)

    # ── API 服务 ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # ── 安全与认证 ──
    admin_username: str = "admin"
    admin_password: str = ""  # 为空时管理后台登录禁用 (默认关闭, 避免硬编码弱口令)
    admin_session_secret: str = ""  # 为空时进程内随机生成 (多 worker 部署必须显式配置)
    admin_session_ttl_hours: int = 12
    admin_cookie_secure: bool = False  # HTTPS 生产环境建议设为 true
    # 允许跨域来源 (JSON 数组); 为空时不启用 CORS (仅同源/反向代理访问)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # ── 速率限制 ──
    rate_limit_enabled: bool = False  # 为 True 时对 /ask /search 按 IP 限流
    rate_limit_rpm: int = 60  # 每 IP 每分钟请求上限 (LLM 相关端点)
    # 管理后台登录防暴力破解: 独立于通用限流, 默认开启 (0=禁用)
    login_rate_limit_rpm: int = 20  # 每 IP 每分钟登录尝试上限

    # ── RAG 检索参数 (可调优, 建议用 tests/eval 评测集回归验证) ──
    # RRF 融合 Dense/BM25 各自的 k 值: k 越小该路排名贡献越大
    rrf_k_dense: int = 60
    rrf_k_sparse: int = 60
    # 查询扩展: 关闭则直接使用原始查询检索 (省一次 LLM 调用)
    query_expansion_enabled: bool = True
    # 多跳检索预算: 控制 LLM 缺口分析轮次与每轮子查询数
    multi_hop_max_rounds: int = 2
    multi_hop_max_sub_queries: int = 3

    # ── 日志 ──
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "logs/app.log"

    @property
    def pre_computed_series_list(self) -> list[int] | None:
        """解析 pre_computed_series 字符串为整数列表."""
        if not self.pre_computed_series:
            return None
        return [int(s.strip()) for s in self.pre_computed_series.split(",") if s.strip()]

    @property
    def pre_computed_abs_path(self) -> Path:
        """预计算数据的绝对路径."""
        p = Path(self.pre_computed_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def documents_abs_dir(self) -> Path:
        """文档源目录的绝对路径."""
        p = Path(self.documents_dir)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def documents_marked_dir(self) -> Path:
        """Markdown 协议数据集目录 (marked/) — 默认嵌入数据源."""
        return self.documents_abs_dir / "marked"

    @property
    def documents_original_dir(self) -> Path:
        """原始 DOCX 文档目录 (original/) — pandoc 处理数据源."""
        return self.documents_abs_dir / "original"

    @property
    def documents_other_dir(self) -> Path:
        """其他文档目录 (other/) — 非 3GPP/O-RAN 文档."""
        return self.documents_abs_dir / "other"

    @property
    def log_abs_file(self) -> Path:
        """日志文件绝对路径."""
        p = Path(self.log_file)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    @property
    def data_abs_dir(self) -> Path:
        """数据根目录绝对路径."""
        p = Path(self.data_dir)
        return p if p.is_absolute() else self.project_root / p

    @property
    def vectors_dir(self) -> Path:
        """向量文件目录."""
        return self.data_abs_dir / "vectors"

    @property
    def bm25_index_path(self) -> Path:
        """BM25 索引文件路径."""
        return self.vectors_dir / "bm25_index.pkl"

    @property
    def manifest_path(self) -> Path:
        """摄入清单文件路径."""
        return self.data_abs_dir / "manifest" / "ingestion_state.json"

    @property
    def embedding_cache_path(self) -> Path:
        """嵌入缓存 SQLite 文件路径."""
        return self.data_abs_dir / "cache" / "embedding_cache.db"

    @property
    def checkpoint_path(self) -> Path:
        """摄入断点 checkpoint 文件路径."""
        return self.data_abs_dir / "checkpoint" / "chunks_checkpoint.pkl"


class IngestionConfig(BaseSettings):
    """摄入管线配置 — 仅重摄入时生效，修改后必须重跑 bulk_ingest.

    .env 中使用 INGESTION__ 前缀覆盖，如 INGESTION__CHUNK_MODE=dynamic
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        env_prefix="INGESTION__",
    )

    # 分块策略
    chunk_mode: Literal["fixed", "dynamic"] = "dynamic"
    chunk_size: int = 1024               # fixed 模式的字符上限 (fallback)
    chunk_overlap: int = 100

    # dynamic 模式 — 内容类型感知上限
    table_max_chars: int = 5000          # 表格 chunk 上限（比正文更大）
    prose_max_chars: int = 1500          # 纯文本 chunk 上限
    max_chunk_chars: int = 8000          # 绝对上限 — BGE-M3 8192 tokens, 留 ~12% 安全边距
                                         # 超限内容强制在最优边界切开，避免嵌入时静默截断

    # Milvus 写入
    batch_size: int = 64


settings = Settings()
ingestion_config = IngestionConfig()
