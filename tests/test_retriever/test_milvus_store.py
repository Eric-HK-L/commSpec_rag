"""milvus_store.py 字段截断工具单元测试.

覆盖 VARCHAR 字段写入 Milvus 前的字节安全截断 (修复 marked 数据集 38.331
深嵌套章节 section_path 2411 字节 > schema 1024 上限导致的入库失败):
  - 短文本原样返回
  - 中文多字节文本按字节截断且不超上限
  - 截断处不产生 UTF-8 解码错误
"""

from src.retriever.milvus_store import _safe_truncate_bytes


class TestSafeTruncateBytes:

    def test_short_text_unchanged(self):
        s = "6.3.1  RRCReconfiguration"
        assert _safe_truncate_bytes(s, 1024) == s

    def test_long_chinese_path_truncated_to_limit(self):
        # 模拟 38.331 深嵌套中文章节路径: 800 字符 × 3 字节 = 2400 字节
        s = "无线资源控制信令流程 > RRC 重配置 > 信令承载配置 > 数据传输承载配置 " * 40
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024
        # 截断后有标记且 UTF-8 完整
        assert out.endswith("…")

    def test_ascii_path_truncated(self):
        s = "a" * 2000
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024

    def test_4096_limit_covers_deepest_path(self):
        # 当前数据集最深 section_path 2411 字节, 4096 上限必须完整容纳
        s = "章节" * 500  # 3000 字节
        out = _safe_truncate_bytes(s, 4096)
        assert len(out.encode("utf-8")) <= 4096
