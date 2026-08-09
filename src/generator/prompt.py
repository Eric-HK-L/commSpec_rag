"""RAG 提示词模板 — 面向 3GPP / O-RAN 规范检索增强生成."""

from __future__ import annotations

import re

from src.retriever.search import RetrievalResult

# ── Grid Table → Pipe Table 转换 ──

# 匹配 Grid Table 边框行: +---+ 或 +===+ 或 +---+---+
_GRID_BORDER = re.compile(r'^\+[+\-=]+\+$')


def _grid_table_to_pipe(text: str) -> str:
    """将 pandoc Grid Table 转换为 GFM pipe table，使前端 ReactMarkdown 可渲染.

    pandoc 默认输出 Grid Table 格式:
        +-------+-------+
        | Col1  | Col2  |
        +=======+=======+
        | val1  | val2  |
        +-------+-------+

    转换为 GFM pipe table:
        | Col1 | Col2 |
        |------|------|
        | val1 | val2 |

    如果文本中没有 Grid Table，原样返回。
    """
    lines = text.split('\n')
    result: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()

        # 检测 Grid Table 起始边框
        if _GRID_BORDER.match(stripped):
            table_start = i
            # 收集整个表格行范围
            i += 1
            while i < n:
                s = lines[i].strip()
                if _GRID_BORDER.match(s):
                    # 判断这是结束边框还是中间边框:
                    # 下一行如果是 |...| → 表格继续; 否则结束
                    next_is_data = (i + 1 < n and lines[i + 1].strip().startswith('|'))
                    if not next_is_data:
                        i += 1
                        break
                i += 1
            table_end = i

            # 转换这个 Grid Table
            pipe_table = _convert_single_grid_table(lines[table_start:table_end])
            if pipe_table:
                result.append(pipe_table)
            continue

        result.append(lines[i])
        i += 1

    return '\n'.join(result)


def _convert_single_grid_table(table_lines: list[str]) -> str:
    """转换单个 Grid Table 为 pipe table."""
    # 提取数据行: 以 | 开头、不是 + 边框的行
    rows: list[list[str]] = []
    header_sep_seen = False
    header_row_idx: int | None = None

    for i, line in enumerate(table_lines):
        stripped = line.strip()
        if stripped.startswith('|'):
            # 提取单元格: 去掉首尾 |，按 | 分割
            cells = [c.strip() for c in stripped[1:-1].split('|')]
            rows.append(cells)
        elif '===' in stripped:
            # +===+ 是表头分隔符，前一行是表头
            header_sep_seen = True
            if rows:
                header_row_idx = len(rows) - 1

    if not rows:
        return ''

    # 确定列数
    ncols = max(len(r) for r in rows) if rows else 0
    if ncols == 0:
        return ''

    # 构建 pipe table
    result_lines: list[str] = []

    # 确定表头行
    if header_sep_seen and header_row_idx is not None:
        header = rows[header_row_idx]
        data_rows = rows[:header_row_idx] + rows[header_row_idx + 1:]
    else:
        # 无显式表头分隔符: 第一行作为表头
        header = rows[0]
        data_rows = rows[1:]

    # 补齐列数
    header = _pad_row(header, ncols)

    # 表头行
    result_lines.append('| ' + ' | '.join(header) + ' |')
    # 分隔行
    result_lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
    # 数据行
    for row in data_rows:
        row = _pad_row(row, ncols)
        result_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(result_lines)


def _pad_row(row: list[str], ncols: int) -> list[str]:
    """补齐行到指定列数."""
    if len(row) < ncols:
        return row + [''] * (ncols - len(row))
    return row[:ncols]


# ── 分类/列举类问题检测 ──

_TAXONOMY_KEYWORDS_CN = [
    "格式", "分类", "种类", "类型", "有哪些", "全部", "所有",
    "列出", "列举", "汇总", "清单", "总结", "多少种",
]
_TAXONOMY_KEYWORDS_EN = [
    "format", "type", "category", "classification", "list all",
    "enumerate", "what are", "kinds of", "summarize", "how many",
]


def is_taxonomy_query(query: str) -> bool:
    """检测查询是否为分类/列举类问题."""
    for kw in _TAXONOMY_KEYWORDS_CN:
        if kw in query:
            return True
    query_lower = query.lower()
    for kw in _TAXONOMY_KEYWORDS_EN:
        if kw in query_lower:
            return True
    return False


_TAXONOMY_EXTRA_PROMPT = """

【分类列举模式】
这是一道分类列举题，请特别注意：
7. 穷举检索片段中涉及的所有格式/类型/分类，不要遗漏任何一个
8. 按层级组织答案：大类→子类→具体条目，使用 Markdown 标题层级（##、###）
9. 如检索片段包含完整表格数据，优先使用 Markdown 管道表格呈现汇总
10. 回答末尾附"覆盖清单"：列出已覆盖的条目名称并标注来源章节号
11. 若检索片段覆盖不全，在末尾明确指出"以下格式未在检索片段中找到"并列出缺失项
12. 对每个列举出的格式/类型，必须从检索片段中提取其关键参数
    （如序列长度、子载波间隔、OFDM符号数、CP长度等），
    以 Markdown 表格逐项呈现，不要只列名称
13. 若条目可按功能/结构特征分为子系列（如 A 系列无 GP、B 系列有 GP），
    按子系列分组说明，并描述各组的设计意图（若检索片段中有相关描述）
14. 若检索片段包含不同大类之间的差异对比信息，在末尾增加对比总结表"""


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    max_context_chunks: int = 10,
    extra_system_note: str = "",
    online_context: str = "",
    history: list[dict[str, str]] | None = None,
    answer_lang: str = "",
) -> list[dict[str, str]]:
    """构建 RAG 问答提示词.

    Args:
        query: 用户查询.
        retrieved_chunks: 检索结果列表.
        max_context_chunks: 最大上下文块数.
        extra_system_note: 可选的额外系统提示 (如 Release 版本说明).
        online_context: 可选的在线搜索补充上下文 (如 Google/TSpec-LLM).
        history: 可选的多轮对话历史 [{"role": "user/assistant", "content": "..."}].
        answer_lang: 期望的回答语言 ('zh'/'ko'/'en'/''=不指定).
            非空时在 user 消息末尾加强输出语言指令 — user 指令比 system 更有效,
            避免 LLM 因英文上下文输出英文再触发回译 (省一次完整生成).
    """
    # 分类列举问题需要更多上下文覆盖 — 从 10 提到 14 (原翻倍到 20,
    # 过大导致 DeepSeek prefill >60s 超时空流; 保守折中保持覆盖同时降延迟)
    _is_taxonomy = is_taxonomy_query(query)
    if _is_taxonomy:
        max_context_chunks = min(14, len(retrieved_chunks))

    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks[:max_context_chunks]):
        # 将 Grid Table 转为 pipe table，确保前端可渲染
        converted_text = _grid_table_to_pipe(chunk.text)
        # 临时覆盖 text 以复用 to_context_str
        original_text = chunk.text
        chunk.text = converted_text
        # 构建引用标签，包含 section_number 层级路径
        ref_label = f"[{i + 1}]"
        section_label = chunk.section_number or chunk.parent_section_id
        section_path = chunk.section_path or chunk.parent_title
        header = f"{ref_label} TS {chunk.spec_number} §{section_label} | {section_path}"
        context_parts.append(f"{header}\n{chunk.text}")

        # 附加相邻 chunk 上下文 — 解决表格/列表内容碎片化问题 (Phase 5 Layer D)
        # 从 4×500 收紧到 2×300: 相邻是 prompt 最大占比项 (实测 69%),
        # 收紧后 token 下降 ~48% 而覆盖语义损失有限 (相邻本为补充而非命中)
        adjacent = getattr(chunk, 'adjacent_chunks', None)
        if adjacent:
            adj_lines = []
            for j, t in enumerate(adjacent[:2]):
                adj_lines.append(f"  [{i+1}.{j+1}] {_grid_table_to_pipe(t)[:300]}")
            if adj_lines:
                context_parts.append("  （相邻上下文：同文档邻近段落, 非检索命中但语义相关）\n" + "\n".join(adj_lines))

        # small-to-big 父章节上下文 — 命中子 chunk 附带所属 section 完整文本 (控制总量)
        # 从 1500 收紧到 800: 父章节仅为理解背景, 截断头部已含章节主旨
        parent_ctx = getattr(chunk, 'parent_context', '')
        if parent_ctx:
            context_parts.append(
                "  （父章节上下文：该片段所属章节的完整文本, 供理解上下文）\n"
                + _grid_table_to_pipe(parent_ctx)[:800]
            )

        chunk.text = original_text
    context_text = "\n\n---\n\n".join(context_parts)

    # 合并在线补充
    full_context = context_text
    if online_context:
        full_context = online_context + "\n\n---\n\n## 离线检索结果 (本地规范库)\n\n" + context_text

    system_prompt = """你是一个通信规范专家助手（3GPP / O-RAN）。回答基于提供的规范文档片段。

回答规则：
1. 严格基于提供的文档片段，不要编造规范内容
2. 每个关键论断必须注明来源 —— 使用内联引用编号 [1][2][3]，对应到最后的 References 表
3. 文档可信度优先级：Physical layer spec 优先（38.211 > 38.212 > 38.213 > 38.214）
   详细说明 > 概述；官方定义 > 一般描述；指定 release 的规范 > 泛用版本
4. 即使检索片段不完整，也应先基于已有片段整理出可确认的信息，再说明缺失项，不要直接放弃回答
5. 【语言要求】必须使用与用户问题相同的语言回答（中文问题用简体中文，英文问题用英文），
   规范术语保留英文原文（如 PDU Session, N2 Interface）；严禁用其他语言输出
6. 回答结构清晰：先给出答案（不要写“直接答案”等字样的标题），再列规范依据
7. 当回答中包含表格时，必须使用 Markdown 管道表格格式（|列1|列2|），禁止使用 Grid Table（+---+）格式
8. 回答末尾附 References 表：每行包含 [编号] Section / Section Hierarchy / Cited Content（原文摘录）/ Relevance
9. 首次引用规范中的技术变量/符号时，必须用中文解释其含义，后续可使用缩写
   示例：\\(\\Delta f_{RA}\\)（随机接入子载波间隔）、\\(\\L_{RA}\\)（前导码序列长度）、\\(\\T_{CP}\\)（循环前缀时长）
16. 片段来源标注了角色标签：🔴权威定义 > 🟡补充参考 > ⚪概述
    优先采纳「🔴权威定义」和「📊参数表」类型的片段作为回答的核心依据"""

    if _is_taxonomy:
        system_prompt += _TAXONOMY_EXTRA_PROMPT

    if extra_system_note:
        system_prompt += "\n\n" + extra_system_note

    user_prompt = f"""## 用户问题

{query}

## 参考规范片段

{full_context}

请基于以上规范片段回答问题。"""

    # 输出语言强指令 — 放 user 消息末尾, 比 system 规则更有效 (DeepSeek 对英文上下文
    # 默认输出英文, 之前导致每次中文提问都触发回译兜底, 多花 ~30s 完整生成)
    _LANG_INSTRUCTION = {
        "zh": "\n\n【输出语言】必须使用简体中文回答，专业术语保留英文原文（如 PDU Session, N2 Interface）。",
        "ko": "\n\n【출력 언어】반드시 한국어로 답변하세요. 전문 용어는 영문 원문을 유지합니다.",
    }
    if answer_lang and answer_lang in _LANG_INSTRUCTION:
        user_prompt += _LANG_INSTRUCTION[answer_lang]

    # 多轮对话：将历史插入最终消息，帮助 LLM 理解追问意图
    if history and len(history) >= 2:
        history_text = "## 对话历史\n\n" + "\n".join(
            f"{'👤 用户' if h['role'] == 'user' else '🤖 助手'}: {h['content'][:300]}"
            for h in history[-8:]
        )
        user_prompt = history_text + "\n\n" + user_prompt

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_query_expansion_prompt(query: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """构建查询扩展提示词 — 支持多轮对话上下文."""
    system_prompt = """你是一个通信规范查询优化器。将用户的自然语言问题转化为更适合检索的关键词组合。

规则：
1. 提取核心协议术语和规范编号
2. 添加同义词和相关协议名
3. 保留关键技术缩写（如 AMF, SMF, gNB, NG-RAN）
4. 输出格式：只输出优化后的查询文本，不要解释"""

    user_content = query
    if history and len(history) >= 2:
        # 将最近几轮对话作为上下文，帮助理解指代消解
        history_lines = ["## 对话历史 (用于理解当前问题的上下文)"]
        for msg in history[-6:]:  # 最近 3 轮 (6 条消息)
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}: {msg['content'][:200]}")
        history_lines.append(f"\n## 当前问题 (需要优化为检索关键词)\n{query}")
        user_content = "\n".join(history_lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
