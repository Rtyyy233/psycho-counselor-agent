"""
Analyst Agent - Deep analysis using retrieval modules.

Works as a background observer. When triggered, analyzes recent conversation
and retrieves relevant memories. Returns analysis string for injection.
"""

from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from mem_retrieve_conv_outline import retrieve_conv_outline
from mem_retrieve_diary import retrieve_diary
from mem_retrieve_material import retrieve_materials
from pydantic import BaseModel, Field
from typing import Literal, Optional
import asyncio


base_model = ChatDeepSeek(model="deepseek-chat", temperature=0.2)

SYSTEM_PROMPT = """你是一位专业的心理分析专家。

职责：
- 分析用户表达的情绪、认知模式和行为模式
- 从记忆中检索相关信息进行对比分析
- 生成洞察性的分析报告

当调用检索工具时，针对用户当前的话题和情绪状态进行检索。"""


class AnalystResult(BaseModel):
    insight: str = Field(description="分析洞察")
    direction: str = Field(description="建议的下一步方向")
    relevant_memories: list[str] = Field(
        default_factory=list, description="相关记忆片段"
    )


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

    # Route to appropriate retrieval based on query characteristics
    # This could be enhanced with LLM-based routing
    try:
        # Try diary first for direct emotional content
        diary_results = await retrieve_diary(last_user_msg)

        # Try materials for indirect content
        material_results = await retrieve_materials(last_user_msg)

        # Try conv_outline for professional context
        conv_results = await retrieve_conv_outline(last_user_msg)

        # Synthesize analysis from results
        analysis = synthesize_analysis(
            query=last_user_msg,
            diary=diary_results,
            materials=material_results,
            conv_outline=conv_results,
        )

        return analysis

    except Exception as e:
        return f"分析过程中出现错误: {str(e)}"


def synthesize_analysis(query: str, diary, materials, conv_outline) -> str:
    """
    Synthesize analysis from retrieval results.

    In a full implementation, this would use LLM to synthesize.
    For now, returns a placeholder structure.
    """
    # Count results (simplified)
    diary_count = len(diary) if diary else 0
    material_count = len(materials) if materials else 0
    conv_count = len(conv_outline) if conv_outline else 0

    if diary_count + material_count + conv_count == 0:
        return f"关于「{query}」，暂无相关记忆。"

    insight_parts = []

    if diary_count > 0:
        insight_parts.append(f"在日记中发现{diary_count}条相关记录")

    if material_count > 0:
        insight_parts.append(f"在个人材料中发现{material_count}条相关内容")

    if conv_count > 0:
        insight_parts.append(f"在咨询记录中发现{conv_count}条相关记录")

    insight = "；".join(insight_parts)

    direction = "可以考虑进一步探索这些记忆，了解背后的模式。"

    return f"{insight}。{direction}"
