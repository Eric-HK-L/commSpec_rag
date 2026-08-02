"""DOCX → Markdown 转换器 — 基于 pandoc（同时解决表格+公式+图片）.

与开源项目对标:
  - gpp-RAG-app: 也用 pandoc，但他们的 clean_content() 删除所有表格（糟粕）→ 我们保留
  - SpecPilot: 早期评估过 Docling（PyTorch 依赖重）→ 最终选用 pandoc 零 Python 依赖
  - Chat3GPP: python-docx 手动解析（无公式支持）→ pandoc 原生 OLE→LaTeX

pandoc 优势:
  - Table → Grid Table (行列结构完整保留)
  - OLE Equation → LaTeX math ($\\mathbf{\\rho}$)
  - Image → 文件引用 (media/image1.emf, 非 base64)
  - Heading → 干净的 # ## ### 层级
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """单文档转换结果."""
    source_file: str              # 原始 DOCX 路径
    spec_number: str              # 如 "38.300"
    release: str                  # 如 "R18"
    version: str                  # 如 "18.4.0"
    title: str                    # 文档标题
    markdown: str                 # 转换后的 Markdown
    errors: list[str] = field(default_factory=list)  # 转换过程中的警告/错误


class PandocExtractor:
    """使用 pandoc 将 3GPP/O-RAN DOCX 规范转换为 Markdown.

    pandoc 相比 mammoth 的核心优势:
    - 表格 → Grid Table (行列结构完整, 不会被 chunker 切散)
    - OLE 公式 → LaTeX math ($\\mathbf{\\rho}_{A}^{d}$)
    - 图片 → 文件引用 (后续可接入 VLM 描述)
    - 无转义污染 (mammoth 的 \\. \\( \\) 问题完全避免)
    """

    # 3GPP 文件名模式: 38xxx-xx.docx 或 38xxx-xx_iXX.docx
    # 前三段: series+spec, release_minor, version_or_info
    SPEC_PATTERN = re.compile(r"^(\d{2})(\d{3})-(\d{2})([a-z]\d{2})?", re.IGNORECASE)

    # O-RAN 文件名模式: O-RAN.WG{1-9}.{SPEC}.{num}-R{rev}-v{major}.{minor}.docx
    # 示例: O-RAN.WG1.CUS.0-R003-v11.00.docx → spec=O-RAN.WG1.CUS.0, version=v11.00
    ORAN_PATTERN = re.compile(
        r"^O-RAN\.WG(\d+)\.(\w+(?:\.\w+)*)-(R\d+)-(v\d+\.\d+)",
        re.IGNORECASE,
    )

    # 文档头信息正则 (多种变体):
    #   主格式: TS 38.300 V18.4.0 (2024-09)
    #   备选1: 第一行纯文本 "3GPP TS 36.213 V18.0.0" 无括号日期
    #   备选2: 末尾 "(Release 18)" 补充 Release 信息
    HEADER_TS_RE = re.compile(
        r"(?:3GPP\s+)?(?:TS|TR)\s+(\d{2}\.\d{3})\s+V(\d+\.\d+\.\d+)(?:\s*\((\d{4}-\d{2})\))?",
        re.IGNORECASE,
    )
    # 备用 Release 检测: 文档头末尾的 (Release N)
    RELEASE_LINE_RE = re.compile(r"\(Release\s+(\d+)\)", re.IGNORECASE)

    # 标题提取: 第一行非空显著文本
    TITLE_LINE_RE = re.compile(r"^(?:3GPP|ETSI)\s+(TS|TR)\s+(\d{2}\.\d{3})\s+.*?(?:Specification|Technical\s+Specification)", re.IGNORECASE)

    def __init__(self):
        pass

    @staticmethod
    def _convert_docx(filepath: Path) -> tuple[str, list[str]]:
        """pandoc DOCX → Markdown 转换.

        pandoc 参数:
          --wrap=none: 不自动折行（避免表格被截断）
          --markdown-headings=atx: 使用 # 风格标题
          -t markdown: 输出标准 Markdown（含 Grid Table 和 LaTeX math）

        Returns:
            (markdown_text, warning_messages)
        """
        try:
            result = subprocess.run(
                [
                    "pandoc", str(filepath),
                    "-t", "markdown",
                    "--wrap=none",
                    "--markdown-headings=atx",
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 大文档最多 2 分钟
            )
            if result.returncode != 0:
                raise RuntimeError(f"pandoc 返回非零: {result.stderr[:200]}")
            warnings = [line for line in result.stderr.split("\n") if line.strip()]
            return result.stdout, warnings
        except FileNotFoundError:
            import platform
            _system = platform.system()
            if _system == "Darwin":
                hint = "brew install pandoc"
            elif _system == "Linux":
                hint = "apt-get install pandoc  或  brew install pandoc"
            elif _system == "Windows":
                hint = "winget install pandoc  或  choco install pandoc"
            else:
                hint = "参见 https://pandoc.org/installing.html"
            raise RuntimeError(f"pandoc 未安装。请运行: {hint}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"pandoc 转换超时: {filepath.name}")

    # ── 单文件 ──

    def extract_file(self, filepath: str | Path) -> ExtractionResult:
        """转换单个 DOCX 文件为 Markdown.

        Args:
            filepath: DOCX 文件路径.

        Returns:
            ExtractionResult 含转换后的 markdown 和元数据.
        """
        filepath = Path(filepath)
        logger.info("转换: %s", filepath.name)

        try:
            md_text, warnings = self._convert_docx(filepath)
        except Exception as e:
            logger.error("pandoc 转换失败 %s: %s", filepath.name, e)
            return ExtractionResult(
                source_file=str(filepath),
                spec_number="", release="", version="", title="",
                markdown="",
                errors=[f"转换失败: {e}"],
            )

        # 提取元数据
        spec_number, release, version = self._parse_filename(filepath.name)
        title = self._extract_title(md_text)

        # 后处理
        md_text = self._postprocess(md_text)

        # 文本头补充: 文件名未解析出的字段用 DOCX 内容头补全
        text_spec, text_release, text_version = self._parse_text_header(md_text)
        if not spec_number:
            spec_number = text_spec
        if not release:
            release = text_release
        if not version:
            version = text_version

        return ExtractionResult(
            source_file=str(filepath),
            spec_number=spec_number,
            release=release,
            version=version,
            title=title,
            markdown=md_text,
        )

    # ── 批量转换 ──

    def extract_directory(
        self,
        dirpath: str | Path,
        pattern: str = "*.docx",
    ) -> list[ExtractionResult]:
        """批量转换目录下所有 DOCX 文件.

        Args:
            dirpath: 搜索目录.
            pattern: 文件匹配模式.

        Returns:
            成功 + 失败的 ExtractionResult 列表 (失败条目 markdown 为空).
        """
        dirpath = Path(dirpath)
        files = sorted(dirpath.rglob(pattern))
        logger.info("批量转换 %d 个文件 (目录: %s)", len(files), dirpath)

        results: list[ExtractionResult] = []
        for fp in files:
            result = self.extract_file(fp)
            results.append(result)

        success = sum(1 for r in results if r.markdown)
        logger.info("批量转换完成: %d/%d 成功", success, len(results))
        return results

    # ── 文件名解析 ──

    @staticmethod
    def _parse_filename(filename: str) -> tuple[str, str, str]:
        """从 3GPP DOCX 文件名提取 spec_number.

        3GPP 命名规则: SSSPP-XX[_suffix].docx
        - SSSPP: Series(2位)+SpecNumber(3位), 如 38300 → TS 38.300
        - XX: 版本标识 (如 i00, 60, 9a0), 不表示 3GPP Release
        - Release 信息仅从文档内容头 (V18.0.0 → R18) 可靠提取

        示例:
          38300-60.docx      → spec=38.300
          36322-i00.docx     → spec=36.322
          23503-i40.docx     → spec=23.503
        """
        name = Path(filename).stem.upper()
        digits = "".join(ch for ch in name if ch.isdigit())
        if len(digits) < 5:
            return "", "", ""
        spec = f"{digits[0:2]}.{digits[2:5]}"
        # Release 不从文件名推断 — 文件名中的数字是内部版本号, 非 3GPP Release
        return spec, "", ""

    @staticmethod
    def parse_oran_filename(filename: str) -> tuple[str, str, str]:
        """从 O-RAN DOCX 文件名提取 spec_number 和 version.

        O-RAN 命名规则: O-RAN.WG{d}.{SPEC}.{num}-R{rev}-v{major}.{minor}.docx
        示例:
          O-RAN.WG1.CUS.0-R003-v11.00.docx → spec=O-RAN.WG1.CUS.0, release=v11.00
          O-RAN.WG4.CUS.0-R003-v07.00.docx → spec=O-RAN.WG4.CUS.0, release=v07.00

        Returns:
            (spec_number, release, version) — release 从 v 版本号推断
        """
        stem = Path(filename).stem
        m = PandocExtractor.ORAN_PATTERN.match(stem)
        if not m:
            return "", "", ""
        wg_num = m.group(1)
        interface = m.group(2)
        revision = m.group(3)  # R003
        version = m.group(4)   # v11.00

        # 构建 O-RAN 规范编号
        spec_id = f"O-RAN.WG{wg_num}.{interface}"

        # ORAN 的 release 用版本号表示
        release = version  # v11.00
        return spec_id, release, version

    @staticmethod
    def _parse_text_header(md_text: str) -> tuple[str, str, str]:
        """从 Markdown 文本头部提取 TS 编号、Release 和版本.

        3GPP 规范头格式:
          主: 3GPP TS 38.300 V18.4.0 (2024-09)
          备选: TS 36.213 V18.0.0 (无日期括号)
        V18 → R18, V17 → R17, V16 → R16 等。
        """
        head = md_text[:800]
        spec, release, version = "", "", ""

        # 主匹配: TS/TR 编号 + V版本
        m = PandocExtractor.HEADER_TS_RE.search(head)
        if m:
            spec = m.group(1)
            version = m.group(2)  # 如 "18.4.0"
            if version:
                major = version.split(".")[0]
                if major.isdigit():
                    release = f"R{major}"

        # 备用 Release 检测: (Release 18) 后缀
        if not release:
            rm = PandocExtractor.RELEASE_LINE_RE.search(head)
            if rm:
                release = f"R{rm.group(1)}"

        return spec, release, version

    @staticmethod
    def _extract_title(md_text: str) -> str:
        """提取文档标题 — pandoc 输出格式简洁."""
        # 策略1: pandoc 的第一行通常是规范头信息
        first_lines = md_text.split("\n")[:20]

        # 策略2: 查找 "Technical Specification" 行及其上下文
        for i, line in enumerate(first_lines):
            if "specification" in line.lower() or "technical" in line.lower():
                # 取前后共 3 行作为标题
                context = [line.strip() for line in first_lines[max(0,i-1):i+3] if line.strip()]
                return " — ".join(context[:4])

        # 策略3: pandoc # 标题
        for line in first_lines:
            line = line.strip()
            if line.startswith("#") and len(line) > 3:
                return line.lstrip("#").strip()

        # 策略4: 取前几行非空连接
        head_lines = [line.strip() for line in first_lines if line.strip() and not line.startswith("!")]
        if head_lines:
            return " — ".join(head_lines[:3])
        return ""

    # ── 后处理 ──

    # pandoc 图片有两种格式:
    #   A) 图片在前，Figure 标题在后 (常见于正文中):
    #      ![](media/image3.emf)
    #      Figure 4.2.1-1: Overview model of the RLC sub layer
    #   B) Figure 标题在前，图片在后 (常见于章节末尾):
    #      Figure 6.2.1.3-1: UMD PDU with 5 bit SN (No LI)
    #      ![](media/image9.emf)
    # 关键区分: 真正的 Figure 标题有 "Figure N.M-K:" 格式（冒号），
    # 正文引用如 "Figure X illustrates..." 应保留在 chunk 中
    FIGURE_AFTER_IMAGE_RE = re.compile(
        r'!\[\]\(media/image[^)]+\)(?:\{[^}]+\})?\s*\n+(Figure\s+\d[\d.]*-\d+:\s*[^\n]+)',
        re.MULTILINE,
    )
    FIGURE_BEFORE_IMAGE_RE = re.compile(
        r'^(Figure\s+\d[\d.]*-\d+:\s*[^\n]+)\n+!\[\]\(media/image[^)]+\)(?:\{[^}]+\})?',
        re.MULTILINE,
    )
    # 封面多图行: ![](media/image1.emf) ![](media/image2.emf)
    MULTI_IMAGE_RE = re.compile(r'(?:!\[\]\(media/image[^)]+\)(?:\{[^}]+\})?\s*){2,}')
    # 孤立无标题图片
    BARE_IMAGE_RE = re.compile(r'!\[\]\(media/image[^)]+\)(?:\{[^}]+\})?\s*')

    @staticmethod
    def _postprocess(md_text: str) -> str:
        """Markdown 后处理 — pandoc 特定清理.

        pandoc 产生的冗余内容:
          - TOC 导航链接 [5](#foreword) [text](#anchor) — 对 RAG 无意义
          - 标题属性 {#id .class} — 内部锚点
          - 图片: 保留 Figure 标题供 RAG/VLM 理解，删除无信息 logo/封面图
          - Change history Annex — 版本变更记录，无技术参考价值，裁剪以减小索引
          - 连续空行过多
        """
        # 1. 移除 pandoc TOC 导航链接 — 保留显示文字，去掉 URL
        md_text = re.sub(r'\[([^\]]+?)\]\(#[^)]+\)', r'\1', md_text)

        # 2. 移除标题的 pandoc 属性: {#id .class .other}
        md_text = re.sub(r'\{[#\.][^}]+\}', '', md_text)

        # 3. 图片处理 — 两种 pandoc Figure 格式
        md_text = PandocExtractor.FIGURE_BEFORE_IMAGE_RE.sub(r'[\1]', md_text)
        md_text = PandocExtractor.FIGURE_AFTER_IMAGE_RE.sub(r'[\1]', md_text)
        md_text = PandocExtractor.MULTI_IMAGE_RE.sub('', md_text)
        md_text = PandocExtractor.BARE_IMAGE_RE.sub('', md_text)

        # 4. 清理残留的空图片引用
        md_text = re.sub(r'!\[\]\([^)]*\)\s*', '', md_text)

        # 5. 裁剪 Change history Annex (无技术价值, 仅版本变更记录)
        #    匹配: "#+ Annex B ... Change history" 或 "#+ Annex ... Change history"
        change_history_pos = None
        for pattern in [
            r'^#+\s+Annex\s+\w+\s*\(.*?\):\s*Change\s+history',
            r'^#+\s+Annex\s+\w+:\s*Change\s+history',
        ]:
            m = re.search(pattern, md_text, re.MULTILINE | re.IGNORECASE)
            if m:
                change_history_pos = m.start()
                break

        if change_history_pos:
            # 保留 Annex 标题前的内容, 去掉 Change history 正文
            md_text = md_text[:change_history_pos].rstrip()

        # 6. 压缩连续空行（最多 2 个）
        md_text = re.sub(r'\n{4,}', '\n\n\n', md_text)

        # 7. 移除行尾空格
        md_text = re.sub(r' +\n', '\n', md_text)

        return md_text.strip()


class MarkdownSourceExtractor:
    """Markdown 协议数据集读取器 — 直接读取数据集 raw.md (跳过 pandoc).

    数据来源: HuggingFace 3GPP/O-RAN Markdown 协议数据集 (转换质量优于 pandoc).
    数据集目录结构 (遵循数据集原结构, 上传时保持一致):

      marked/R18/38_series/38865/raw.md                          → spec=38.865, release=R18
      marked/R18/36_series/36108/raw.md                          → spec=36.108, release=R18
      marked/ORAN/O-RAN.WG4.TS.CUS.0-R005-v20.00/raw.md          → spec=O-RAN.WG4.TS.CUS.0, release=v20.00

    元数据解析: 优先从目录路径 (spec 目录名 + series 目录 + release 目录),
    其次从内容头 (3GPP TS/TR 编号 / O-RAN ALLIANCE 标识) 兜底.
    """

    # 规范目录名: 纯 5 位数字 (如 38865) → spec 38.865; 含子规范号 (如 38101-2) → spec 38.101
    SPEC_DIR_RE = re.compile(r"^(\d{5})(?:-\d+)?$")
    # series 目录: 38_series → 38
    SERIES_DIR_RE = re.compile(r"^(\d{2})_series$")
    # Release 目录: R18 → 18
    RELEASE_DIR_RE = re.compile(r"^R(\d+)$", re.IGNORECASE)

    # 内容头 3GPP 编号 (复用 PandocExtractor 的主格式, 但无日期括号也接受)
    HEADER_SPEC_RE = re.compile(r"(?:3GPP\s+)?(?:TS|TR)\s+(\d{2}\.\d{3})\s+V(\d+\.\d+\.\d+)", re.IGNORECASE)
    HEADER_RELEASE_RE = re.compile(r"\(Release\s+(\d+)\)", re.IGNORECASE)
    HEADER_ORAN_RE = re.compile(r"O-RAN\s+ALLIANCE|O-RAN\s+Working\s+Group", re.IGNORECASE)

    def __init__(self):
        pass

    def extract_file(self, filepath: str | Path) -> ExtractionResult:
        """读取单个 raw.md 数据集文件.

        Args:
            filepath: markdown 文件路径.

        Returns:
            ExtractionResult 含 markdown 文本和元数据 (spec_number/release/version/title).
        """
        filepath = Path(filepath)
        logger.info("读取 Markdown 数据集: %s", filepath)

        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("读取失败 %s: %s", filepath.name, e)
            return ExtractionResult(
                source_file=str(filepath),
                spec_number="", release="", version="", title="",
                markdown="",
                errors=[f"读取失败: {e}"],
            )

        # 路径解析 (目录名优先)
        spec_number, release, version, doc_type = self._parse_path(filepath)

        # 内容头兜底 (spec/release/version)
        head = text[:800]
        m = self.HEADER_SPEC_RE.search(head)
        if m:
            if not spec_number:
                spec_number = m.group(1)
            if not version:
                version = m.group(2)
            if not release and version:
                major = version.split(".")[0]
                if major.isdigit():
                    release = f"R{major}"
        if not release:
            rm = self.HEADER_RELEASE_RE.search(head)
            if rm:
                release = f"R{rm.group(1)}"
        if not doc_type and self.HEADER_ORAN_RE.search(head):
            doc_type = "oran"

        title = self._extract_title(text)
        clean = self._clean_markdown(text)

        return ExtractionResult(
            source_file=str(filepath),
            spec_number=spec_number,
            release=release,
            version=version,
            title=title,
            markdown=clean,
        )

    def extract_directory(self, dirpath: str | Path, pattern: str = "*.md") -> list[ExtractionResult]:
        """批量读取目录下所有 markdown 数据集文件."""
        dirpath = Path(dirpath)
        files = sorted(dirpath.rglob(pattern))
        logger.info("批量读取 %d 个 markdown 文件 (目录: %s)", len(files), dirpath)

        results: list[ExtractionResult] = []
        for fp in files:
            result = self.extract_file(fp)
            results.append(result)

        success = sum(1 for r in results if r.markdown)
        logger.info("批量读取完成: %d/%d 成功", success, len(results))
        return results

    # ── 路径元数据解析 ──

    @staticmethod
    def _parse_path(filepath: Path) -> tuple[str, str, str, str]:
        """从数据集目录路径解析元数据.

        Returns:
            (spec_number, release, version, doc_type) — 无法解析的字段为空字符串.
        """
        spec_number, release, version, doc_type = "", "", "", ""
        parts = filepath.parts
        # 找标记目录名: 5 位数字规范目录 或 O-RAN.* 目录
        for i, part in enumerate(parts):
            m = MarkdownSourceExtractor.SPEC_DIR_RE.match(part)
            if m and not spec_number:
                digits = m.group(1)
                spec_number = f"{digits[:2]}.{digits[2:]}"
                doc_type = "3gpp"
                continue
            if part.upper().startswith("O-RAN.") and not spec_number:
                # 注意: 不能走 parse_oran_filename — 其内部 Path().stem 会把 "v20.00" 的
                # ".00" 当扩展名剥离, 导致匹配失败。这里直接用 pattern 匹配目录名。
                m = PandocExtractor.ORAN_PATTERN.match(part)
                if m:
                    spec_number = f"O-RAN.WG{m.group(1)}.{m.group(2)}"
                    version = m.group(4)
                    release = version  # O-RAN 以版本号作为 release 标识
                    doc_type = "oran"
                continue
            # Release 目录 (R18) — 3GPP 才使用
            rm = MarkdownSourceExtractor.RELEASE_DIR_RE.match(part)
            if rm and not release:
                release = f"R{rm.group(1)}"
                continue
            sm = MarkdownSourceExtractor.SERIES_DIR_RE.match(part)
            if sm and not spec_number:
                # 36_series 目录出现但无规范目录 → 无法推断具体 spec
                continue
        return spec_number, release, version, doc_type

    @staticmethod
    def _extract_title(md_text: str) -> str:
        """提取数据集文档标题 — 优先规范头行, 其次首个标题."""
        lines = md_text.split("\n")
        # 1. "3GPP TS/TR 38.xxx V18.0.0 ..." 行
        for line in lines[:30]:
            s = line.strip().lstrip("#").strip()
            if re.search(r"3GPP\s+(TS|TR)\s+\d{2}\.\d{3}\s+V\d+", s, re.IGNORECASE):
                return s
        # 2. O-RAN 标题行
        for line in lines[:30]:
            s = line.strip().lstrip("#").strip()
            if s.startswith("O-RAN Working Group") or s.startswith("O-RAN ALLIANCE"):
                return s
        # 3. 首个非 Contents 标题
        for line in lines[:60]:
            s = line.strip()
            if s.startswith("#") and len(s) > 3 and "contents" not in s.lower() and "logo" not in s.lower():
                return s.lstrip("#").strip()
        return ""

    # ── Markdown 清理 ──

    @staticmethod
    def _clean_markdown(md_text: str) -> str:
        """数据集 markdown 清理 — 移除图片引用与噪声.

        数据集特点:
          - 3GPP raw.md: ![5G Advanced logo](xxx_img.jpg) 图片引用 (文件多数缺失)
          - O-RAN raw.md: <img src="retrieval/oran/..." /> HTML 图片标签
          - TOC 导航链接 [text](#anchor) — 对 RAG 无意义
        """
        # 1. HTML img 标签 (O-RAN 数据集)
        md_text = re.sub(r"<img\s+[^>]*/?>", "", md_text, flags=re.IGNORECASE)

        # 2. Markdown 图片引用 ![alt](path) → 保留 alt 文字 (图片标题即图注信息)
        #    ![5G Advanced logo](xxx_img.jpg) → [图: 5G Advanced logo]
        #    空 alt 的 ![](path) 直接删除
        def _keep_alt(m: re.Match) -> str:
            alt = m.group(1).strip()
            return f"[图: {alt}]" if alt else ""

        md_text = re.sub(r"!\[([^\]]*)\]\([^)]*\)\s*", _keep_alt, md_text)

        # 3. 锚点导航链接 [text](#anchor) → 保留文字
        md_text = re.sub(r"\[([^\]]+?)\]\(#[^)]+\)", r"\1", md_text)

        # 4. 压缩连续空行 (最多 2 个)
        md_text = re.sub(r"\n{4,}", "\n\n\n", md_text)

        # 5. 移除行尾空格
        md_text = re.sub(r" +\n", "\n", md_text)

        return md_text.strip()
