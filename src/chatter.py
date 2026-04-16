"""
Chatter Agent - Conversational interface for psychological counselor.

User sees: warm, empathetic conversation
Behind the scenes: observes context, incorporates analyst/supervisor injections
"""
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional
import asyncio


base_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.5,
)

SYSTEM_PROMPT = """你是一位温柔、有同理心的心理咨询陪伴者。

角色特点：
- 倾听为主，不急于给建议
- 适时共情，温和引导
- 注意用户的情绪变化，给予空间

当有[分析洞察]或[指导建议]时，将其自然地融入你的回复中，但不要直接提及这些标签。"""


class ChatterResponse(BaseModel):
    reply: str = Field(description="回复内容")
    analysist_trigger: Literal["false", "true"] = Field(
        default="false",
        description="是否触发深度分析"
    )


async def call_chatter(ctx) -> str:
    """
    Chatter generates response using context + observer injections.

    Args:
        ctx: SharedContext instance with messages and pending injections

    Returns:
        str: Chatter's response text
    """
    chatter = base_model.with_structured_output(ChatterResponse)

    # Build messages for LLM
    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Read pending injections
    analyst_injection = ctx.analyst_injection
    supervisor_injection = ctx.supervisor_injection

    # Build enriched system prompt with injections
    enriched_prompt = SYSTEM_PROMPT
    if analyst_injection:
        enriched_prompt += f"\n\n[分析洞察]\n{analyst_injection.content}"
        ctx.analyst_injection = None  # Clear after use

    if supervisor_injection:
        enriched_prompt += f"\n\n[指导建议]\n{supervisor_injection.content}"
        ctx.supervisor_injection = None  # Clear after use

    llm_messages[0] = SystemMessage(content=enriched_prompt)

    # Add conversation history
    for msg in ctx.messages[-10:]:  # Last 10 messages
        if msg.role == "user":
            llm_messages.append(HumanMessage(content=msg.content))
        else:
            llm_messages.append(AIMessage(content=msg.content))

    # Generate response
    result = await chatter.ainvoke(llm_messages)
    reply = result.reply if hasattr(result, 'reply') else result.get('reply', '')

    return reply