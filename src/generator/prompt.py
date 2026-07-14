"""RAG 提示词模板 — 面向 3GPP 规范检索增强生成."""

from __future__ import annotations

from src.retriever.search import RetrievalResult


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    max_context_chunks: int = 10,
    extra_system_note: str = "",
    online_context: str = "",
) -> list[dict[str, str]]:
    """构建 RAG 问答提示词.

    Args:
        query: 用户查询.
        retrieved_chunks: 检索结果列表.
        max_context_chunks: 最大上下文块数.
        extra_system_note: 可选的额外系统提示 (如 Release 版本说明).
        online_context: 可选的在线搜索补充上下文 (如 Google/TSpec-LLM).
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(retrieved_chunks[:max_context_chunks]):
        context_parts.append(chunk.to_context_str(i))
    context_text = "\n\n---\n\n".join(context_parts)

    # 合并在线补充
    full_context = context_text
    if online_context:
        full_context = online_context + "\n\n---\n\n## 离线检索结果 (本地规范库)\n\n" + context_text

    system_prompt = """你是一个 3GPP 规范专家助手。回答基于提供的 3GPP 规范文档片段。

回答规则：
1. 严格基于提供的文档片段，不要编造规范内容
2. 每个关键论断必须注明来源（引用的 TS 编号和章节号）
3. 如果文档片段不足以回答，明确说明"根据提供的规范片段无法确定"
4. 使用中文回答，但保留规范术语的英文原文（如 PDU Session, N2 Interface）
5. 回答结构清晰：先给出直接答案，再列出规范依据"""

    if extra_system_note:
        system_prompt += "\n\n" + extra_system_note

    user_prompt = f"""## 用户问题

{query}

## 参考规范片段

{full_context}

请基于以上规范片段回答问题。"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_query_expansion_prompt(query: str) -> list[dict[str, str]]:
    """构建查询扩展提示词."""
    system_prompt = """你是一个 3GPP 规范查询优化器。将用户的自然语言问题转化为更适合检索的关键词组合。

规则：
1. 提取核心协议术语和规范编号
2. 添加同义词和相关协议名
3. 保留关键技术缩写（如 AMF, SMF, gNB, NG-RAN）
4. 输出格式：只输出优化后的查询文本，不要解释"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
