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


class TestDefaults:
    """默认值验证."""

    def test_log_level_default(self):
        s = Settings()
        assert s.log_level == "INFO"

    def test_llm_defaults(self):
        s = Settings()
        assert s.llm_temperature == 0.0
        assert s.llm_max_tokens == 2048
        assert s.llm_timeout == 60.0

    def test_retrieval_defaults(self):
        s = Settings()
        assert s.max_search_results == 10
        assert s.dense_top_k == 100
        assert s.similarity_threshold == 0.7

    def test_vector_db_default(self):
        s = Settings()
        assert s.vector_db == "milvus"
        assert s.milvus_port == 19530
