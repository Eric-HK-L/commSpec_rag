"""settings.py 单元测试 — 计算属性与路径解析."""

from pathlib import Path

from src.config.settings import Settings


class TestPreComputedSeriesList:
    """pre_computed_series_list — 系列号字符串解析."""

    def test_empty_string(self):
        s = Settings(pre_computed_series="")
        assert s.pre_computed_series_list is None

    def test_single_series(self):
        s = Settings(pre_computed_series="38")
        assert s.pre_computed_series_list == [38]

    def test_multiple_series(self):
        s = Settings(pre_computed_series="21,22,23,24")
        assert s.pre_computed_series_list == [21, 22, 23, 24]

    def test_whitespace(self):
        s = Settings(pre_computed_series=" 38 , 21 ")
        assert s.pre_computed_series_list == [38, 21]

    def test_empty_parts(self):
        s = Settings(pre_computed_series="38,,21")
        assert s.pre_computed_series_list == [38, 21]


class TestPathProperties:
    """绝对路径计算属性."""

    def test_pre_computed_abs_path_relative(self):
        s = Settings(pre_computed_path="data/3GPP-R18")
        p = s.pre_computed_abs_path
        assert isinstance(p, Path)
        assert p.is_absolute()

    def test_pre_computed_abs_path_absolute(self):
        s = Settings(pre_computed_path="/tmp/3gpp")
        assert s.pre_computed_abs_path == Path("/tmp/3gpp")

    def test_documents_abs_dir_default(self):
        s = Settings()
        p = s.documents_abs_dir
        assert isinstance(p, Path)
        assert p.is_absolute()

    def test_log_abs_file(self):
        s = Settings(log_file="logs/test.log")
        p = s.log_abs_file
        assert isinstance(p, Path)
        assert p.is_absolute()


class TestResolvedEmbeddingDevice:
    """resolved_embedding_device — 设备解析."""

    def test_explicit_cpu(self):
        s = Settings(embedding_device="cpu")
        assert s.resolved_embedding_device == "cpu"

    def test_explicit_cuda(self):
        s = Settings(embedding_device="cuda")
        assert s.resolved_embedding_device == "cuda"

    def test_auto_detection(self):
        s = Settings(embedding_device="auto")
        device = s.resolved_embedding_device
        assert device in ("cpu", "cuda", "mps")


class TestApiKeyResolution:
    """LLM API key 解析 — 显式值优先, 空/占位符时从密钥文件回退读取.

    场景: .env 不再保存明文密钥, 生产密钥存放于独立文件 (默认 ~/ds-api-key).
    """

    def test_explicit_key_wins(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("file-key")
        s = Settings(llm_api_key="explicit-key", llm_api_key_file=str(key_file))
        assert s.llm_api_key == "explicit-key"

    def test_reads_from_file_when_empty(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("file-key\n")
        s = Settings(llm_api_key="", llm_api_key_file=str(key_file))
        assert s.llm_api_key == "file-key"

    def test_placeholder_replaced_by_file(self, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("file-key")
        s = Settings(llm_api_key="sk-your-key-here", llm_api_key_file=str(key_file))
        assert s.llm_api_key == "file-key"

    def test_missing_file_keeps_empty(self, tmp_path):
        s = Settings(llm_api_key="", llm_api_key_file=str(tmp_path / "nope"))
        assert s.llm_api_key == ""


class TestDefaults:
    """默认值验证 — 直接断言声明默认值, 免疫 .env/环境变量覆盖.

    注: pymilvus 等依赖导入时会把 .env 注入 os.environ,
    用 Settings() 构造验证默认值会受环境污染, 故直接检查字段声明.
    """

    @staticmethod
    def _default(name: str):
        return Settings.model_fields[name].default

    def test_log_level_default(self):
        assert self._default("log_level") == "INFO"

    def test_llm_defaults(self):
        assert self._default("llm_temperature") == 0.0
        assert self._default("llm_max_tokens") == 2048
        assert self._default("llm_timeout") == 60.0

    def test_retrieval_defaults(self):
        assert self._default("max_search_results") == 20  # 对比类问题需要更多候选覆盖多规范
        assert self._default("dense_top_k") == 100

    def test_vector_db_default(self):
        assert self._default("vector_db") == "milvus"
        assert self._default("milvus_port") == 19530
