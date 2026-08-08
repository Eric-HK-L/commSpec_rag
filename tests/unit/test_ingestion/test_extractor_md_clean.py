"""extractor.py MarkdownSourceExtractor 清洗逻辑单元测试.

覆盖 _clean_markdown 的图片引用处理 (方案 A):
  - ![alt](path) → [图: alt] 保留图片标题
  - 空 alt 的 ![](path) 直接删除
  - <img .../> HTML 标签删除 (O-RAN 数据集)
  - 锚点导航链接 [text](#anchor) 保留文字
"""

from src.ingestion.extractor import MarkdownSourceExtractor


class TestCleanMarkdownImages:
    """图片引用处理."""

    def test_keeps_alt_text(self):
        md = "before\n![5G Advanced logo](xxx_img.jpg)\nafter"
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "[图: 5G Advanced logo]" in out
        assert "xxx_img.jpg" not in out
        assert "![" not in out

    def test_keeps_alt_with_figure_number(self):
        md = "![Figure 4.2.1-1: Architecture](img/fig421.png)"
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "[图: Figure 4.2.1-1: Architecture]" in out

    def test_empty_alt_removed(self):
        md = "a\n![](path/to/img.jpg)\nb"
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "![" not in out
        assert "[图:" not in out

    def test_html_img_tag_removed(self):
        md = 'before<img src="retrieval/oran/logo.png"/>after'
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "<img" not in out
        assert "retrieval/oran" not in out
        assert "before" in out and "after" in out

    def test_anchor_link_keeps_text(self):
        md = "see [Section 6.1](#6-1) for details"
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "Section 6.1" in out
        assert "#6-1" not in out

    def test_blank_lines_collapsed(self):
        md = "a\n\n\n\n\nb"
        out = MarkdownSourceExtractor._clean_markdown(md)
        assert "\n\n\n\n" not in out
