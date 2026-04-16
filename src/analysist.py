"""
Analyst Agent - Deep analysis using retrieval modules.

Works as a background observer. When triggered, analyzes recent conversation
and retrieves relevant memories. Returns analysis string for injection.
"""

from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from mem_retrieve_conv_outline import retrieve_conv_outline
from mem_retrieve_diary import retrieve_diary
from mem_retrieve_material import retrieve_materials
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import asyncio


# Create an agent with retrieval tools
SYSTEM_PROMPT = """你是一位专业的心理分析专家。

职责：
- 分析用户表达的情绪、认知模式和行为模式
- 从记忆中检索相关信息进行对比分析
- 生成洞察性的分析报告

工具说明：
- retrieve_diary: 从日记中检索相关信息
- retrieve_materials: 从个人材料中检索相关信息
- retrieve_conv_outline: 从咨询记录中检索相关信息

工作流程：
1. 分析用户最近的对话内容，理解当前话题和情绪状态
2. 根据分析决定需要检索哪些记忆类型
3. 调用相应的检索工具获取相关信息
4. 综合分析检索结果，生成洞察性报告
5. 输出包含分析洞察、相关记忆和建议方向的结构化报告

请使用合适的工具进行检索，然后综合分析结果。"""

# Create LLM for analysis
base_model = ChatDeepSeek(model="deepseek-chat", temperature=0.2)

# Create agent with retrieval tools
analyst_agent = create_agent(
    model=base_model,
    tools=[retrieve_diary, retrieve_materials, retrieve_conv_outline],
    system_prompt=SYSTEM_PROMPT,
)


class AnalystResult(BaseModel):
    insight: str = Field(description="分析洞察")
    direction: str = Field(description="建议的下一步方向")
    relevant_memories: list[str] = Field(
        default_factory=list, description="相关记忆片段"
    )


async def synthesize_analysis(
    query: str, diary_results, material_results, conv_results
) -> str:
    """Synthesize analysis from retrieval results."""
    # Collect relevant texts
    all_texts = []

    for result in diary_results:
        for doc in result.documents:
            all_texts.append(f"[日记] {doc.page_content[:500]}")  # Limit length

    for result in material_results:
        for doc in result.documents:
            all_texts.append(f"[材料] {doc.page_content[:500]}")

    for result in conv_results:
        for doc in result.documents:
            all_texts.append(f"[咨询记录] {doc.page_content[:500]}")

    if not all_texts:
        return "暂无相关记忆可供分析。"

    # Prepare context for LLM
    context = "\n\n".join(all_texts[:10])  # Limit to 10 snippets

    from langchain_deepseek import ChatDeepSeek
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)

    prompt = f"""基于以下相关记忆，分析用户的查询并提供专业心理分析：

用户查询：{query}

相关记忆片段：
{context}

请提供：
1. 对用户当前情绪状态和认知模式的分析
2. 记忆片段中发现的模式和关联
3. 建议的下一步探索方向

请用中文回复，保持专业且富有洞察力。"""

    response = await llm.ainvoke(
        [
            SystemMessage(
                content="你是一位专业的心理分析专家。根据提供的记忆片段进行分析。"
            ),
            HumanMessage(content=prompt),
        ]
    )

    if hasattr(response, "content"):
        return response.content
    else:
        return str(response)


async def call_analysist(ctx) -> Optional[str]:
    """
    Analyst performs deep analysis based on recent conversation.

    Args:
        ctx: SharedContext with messages

    Returns:
        str: Analysis result for injection, or None if not triggered
    """
    # Get recent messages for context
    recent = await ctx.get_recent_messages(5)
    if len(recent) < 2:
        return None

    # Build query from recent user message
    last_user_msg = next((m.content for m in reversed(recent) if m.role == "user"), "")

    # Use direct retrieval instead of agent to avoid LangGraph issues
    try:
        # Run retrievals in parallel using tool's ainvoke method
        diary_results, material_results, conv_results = await asyncio.gather(
            retrieve_diary.ainvoke({"query": last_user_msg}),
            retrieve_materials.ainvoke({"query": last_user_msg}),
            retrieve_conv_outline.ainvoke({"query": last_user_msg}),
            return_exceptions=True,  # Don't fail if one retrieval fails
        )

        # Convert exceptions to empty results
        if isinstance(diary_results, Exception):
            print(f"Diary retrieval error: {diary_results}")
            diary_results = []
        if isinstance(material_results, Exception):
            print(f"Material retrieval error: {material_results}")
            material_results = []
        if isinstance(conv_results, Exception):
            print(f"Conversation retrieval error: {conv_results}")
            conv_results = []

        # Synthesize analysis
        analysis = await synthesize_analysis(
            query=last_user_msg,
            diary_results=diary_results,
            material_results=material_results,
            conv_results=conv_results,
        )

        # Ensure we have meaningful analysis
        if not analysis or len(analysis.strip()) < 10:
            return None

        return analysis

    except Exception as e:
        print(f"Analyst error: {e}")
        # Fallback to simple analysis without retrieval
        try:
            from langchain_deepseek import ChatDeepSeek
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
            prompt = f"请分析用户的以下表达，提供简要的心理分析：{last_user_msg}"

            response = await llm.ainvoke(
                [
                    SystemMessage(content="你是一位心理分析专家。"),
                    HumanMessage(content=prompt),
                ]
            )

            if hasattr(response, "content"):
                return response.content[:500]  # Limit length
            else:
                return str(response)[:500]
        except Exception as inner_e:
            return f"分析失败: {str(inner_e)}"
