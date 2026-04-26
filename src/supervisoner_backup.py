"""
Supervisor Agent - Clinical oversight for conversation flow.

Observes conversation and injects guidance when:
- Chatter is about to change topic but there's an unresolved theme
- Emotional depth is detected but Chatter is moving too fast
- Silence/hesitation detected
- Therapeutic direction needs adjustment
"""

from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional
from config import LLM_MODEL


base_model = ChatDeepSeek(model=LLM_MODEL, temperature=0.2)

SYSTEM_PROMPT = """你是一位临床督导专家，观察咨询对话并提供指导。

你的职责：
- 判断对话节奏是否合适
- 检测是否有时机被错过
- 识别用户是否在逃避某个话题
- 判断是否需要更深层的探索

当发现问题时，生成简短、具体的指导建议。

输出格式：
- guidance: 具体指导内容（1-2句话）
- priority: high/medium/low
- reason: 判断理由"""


class SupervisorResult(BaseModel):
    guidance: str = Field(description="指导建议")
    priority: Literal["high", "medium", "low"] = Field(default="medium")
    reason: str = Field(description="判断理由")
    should_inject: bool = Field(description="是否需要注入")


async def supervisoner(messages: list[dict], context: dict) -> Optional[str]:
    """
    Supervisor evaluates conversation state and returns guidance.

    Args:
        messages: Recent conversation messages
        context: Additional context including current_topic

    Returns:
        str: Guidance string for injection, or None
    """
    if not messages:
        return None

    last_msg = messages[-1].get("content", "") if messages else ""

    # Check for deep emotional content (can work with 1 message)
    emotional_markers = ["难过", "痛苦", "害怕", "崩溃", "哭", "伤心"]
    has_emotion = any(marker in last_msg for marker in emotional_markers)

    if has_emotion:
        return "检测到强烈情绪，这是一个值得深挖的时机。"

    if len(messages) < 2:
        return None

    # Check for avoidance patterns
    avoidance_markers = ["算了", "不说这个了", "跳过", "没什么", "就这样吧"]
    if any(marker in last_msg for marker in avoidance_markers):
        return "用户似乎在回避话题，可以温和地邀请继续分享。"

    # Check topic change patterns
    current_topic = context.get("current_topic", "")
    if current_topic and len(messages) >= 4:
        prev_topics = [m.get("content", "")[:30] for m in messages[-4:-2]]
        # Simple topic change detection
        topic_change = any(current_topic[:20] not in p for p in prev_topics)
        if topic_change:
            return "话题发生转换，注意是否有未完成的探索。"

    # No specific guidance needed
    return None


async def supervisoner_ainvoke(input: dict) -> Optional[str]:
    """
    Wrapper for supervisor invocation - compatible with top_module call pattern.
    """
    messages = input.get("messages", [])
    ctx = input.get("context", {})

    return await supervisoner(messages, ctx)
