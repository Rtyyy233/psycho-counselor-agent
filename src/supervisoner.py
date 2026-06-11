"""
Supervisor Agent - Clinical oversight for conversation flow.

Observes conversation and injects guidance when:
- Chatter is about to change topic but there's an unresolved theme
- Emotional depth is detected but Chatter is moving too fast
- Silence/hesitation detected
- Therapeutic direction needs adjustment
"""

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional
from config import LLM_MODEL


# Create LLM for supervision
base_model = ChatDeepSeek(
    model=LLM_MODEL,
    temperature=0.1,  # Lower temperature for more consistent supervision
)

SYSTEM_PROMPT = """你是一位临床督导专家，基于Falender & Shafranske督导框架和Sommers-Flanagan临床面谈框架，观察心理咨询对话并提供指导。

你的职责：
- 分析对话节奏和深度是否合适
- 检测是否有时机被错过（如情绪表达、重要话题出现）
- 识别用户是否在逃避某个话题或需要更多支持
- 判断是否需要更深层的探索或调整对话方向
- 评估对话的整体治疗进展

分析重点（基于三本核心教材框架）：
1. 情绪深度：用户是否表达了深层情绪，对话是否充分探索了这些情绪
2. 话题连续性：话题转换是否自然，是否有未完成的话题需要返回
3. 回避模式：用户是否表现出回避、防御或不愿意深入讨论
4. 治疗时机：是否有需要立即关注的危机信号或重要治疗机会
5. 对话节奏：对话节奏是否太快或太慢，是否给了用户足够的时间表达
6. 治疗联盟状态：是否存在联盟破裂信号——退缩型（沉默、表面顺从、疏离）或对抗型（不满、质疑）——如果检测到，建议使用元沟通技术修复
7. 反移情信号：Chatter是否陷入过度建议、过度认同、过度疏离或回避话题——如果检测到，提醒Chatter自我觉察
8. 贯注行为质量（Sommers-Flanagan）：Chatter是否避免了消极贯注行为（过多点头、过多嗯哼、过多目光接触、重复最后一个词、笨拙模仿）
9. 提问技术平衡（Sommers-Flanagan）：开放式vs封闭式问题的比例是否合适，是否过度使用为什么问题
10. 危机信号（Sommers-Flanagan自杀评估）：来访者是否表达了自杀意念/计划/手段、突然平静（可能是自杀决心已定的信号）、情绪严重失调

输出格式要求：
- 如果没有需要干预的问题，返回"无指导需求"
- 如果有需要干预的问题，返回简洁的指导建议（1-2句话）
- 指导建议应该具体、可操作，帮助Chatter改善对话质量

示例：
用户："我最近感到非常焦虑，晚上睡不着觉。"
指导："这是一个探索焦虑根源的好时机，可以询问更多关于焦虑的具体表现和触发因素。"

用户："算了，不说这个了，我们聊点别的吧。"
指导："用户表现出回避，可以温和地邀请继续分享，同时表达理解和支持。"

用户："我昨天和同事吵架了，感觉很糟糕。"
指导："关注用户的情绪体验，探索吵架事件的具体细节和用户的感受。"

请基于对话内容进行专业判断，只在实际需要干预时才提供指导。"""


class SupervisorResult(BaseModel):
    guidance: str = Field(description="指导建议，如果没有指导需求则为空字符串")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium", description="指导优先级"
    )
    reason: str = Field(description="判断理由")
    should_inject: bool = Field(description="是否需要注入指导")


# Create structured output supervisor
supervisor_llm = base_model.with_structured_output(SupervisorResult)


async def supervisoner(messages: list[dict], context: dict) -> Optional[str]:
    """
    Supervisor evaluates conversation state and returns guidance using LLM.

    Args:
        messages: Recent conversation messages
        context: Additional context including current_topic

    Returns:
        str: Guidance string for injection, or None if no guidance needed
    """
    if not messages or len(messages) < 2:
        return None

    # Format conversation for analysis
    conversation_text = "\n".join(
        [
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages[-5:]  # Last 5 messages for context
        ]
    )

    current_topic = context.get("current_topic", "")

    # Create analysis prompt
    analysis_prompt = f"""请分析以下心理咨询对话，判断是否需要提供督导指导：

当前对话主题：{current_topic}

最近对话记录：
{conversation_text}

请分析：
1. 对话的节奏和深度是否合适
2. 是否有重要的情绪表达需要更深入探索
3. 用户是否表现出回避或防御
4. 话题转换是否自然，是否有未完成的话题
5. 是否有需要立即关注的危机信号或治疗机会

基于以上分析，请提供督导指导（如果有需要的话）。"""

    try:
        # Get structured analysis from LLM
        result: SupervisorResult = await supervisor_llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=analysis_prompt),
            ]
        )

        # Check if guidance is needed
        if result.should_inject and result.guidance and result.guidance.strip():
            # Return guidance string (combine guidance and reason for context)
            return f"{result.guidance} [理由：{result.reason}]"
        else:
            return None

    except Exception as e:
        print(f"Supervisor LLM error: {e}")
        # Fallback to simple heuristic if LLM fails
        return await _fallback_supervisor(messages, context)


async def _fallback_supervisor(messages: list[dict], context: dict) -> Optional[str]:
    """
    Fallback supervisor using simple heuristics if LLM fails.
    """
    if not messages:
        return None

    # Check for avoidance patterns in recent messages (last 3)
    avoidance_markers = [
        "算了",
        "不说这个了",
        "跳过",
        "没什么",
        "就这样吧",
        "不想说了",
        "换个话题",
    ]
    for msg in messages[-3:]:
        content = msg.get("content", "")
        if any(marker in content for marker in avoidance_markers):
            return "用户似乎在回避话题，可以温和地邀请继续分享。同时留意这是否为联盟破裂的退缩型信号。"

    # Check all recent messages (last 5) for emotional content
    emotional_markers = [
        "难过",
        "痛苦",
        "害怕",
        "崩溃",
        "哭",
        "伤心",
        "焦虑",
        "抑郁",
        "绝望",
    ]
    # Check each message content for emotional markers
    for msg in messages[-5:]:
        content = msg.get("content", "")
        if any(marker in content for marker in emotional_markers):
            return "检测到强烈情绪，这是一个值得深挖的时机。"

    # Check for crisis/suicide signals (Sommers-Flanagan framework)
    crisis_markers = [
        "不想活",
        "自杀",
        "结束生命",
        "死",
        "活不下去",
        "没有意义",
        "消失",
        "离开这个世界",
    ]
    for msg in messages[-5:]:
        content = msg.get("content", "")
        if any(marker in content for marker in crisis_markers):
            return "【高优先级】检测到可能的危机信号，建议直接但温和地询问自杀意念。参考Sommers-Flanagan：直接询问不会植入这个想法，评估计划、手段和时机。"

    # Check for alliance rupture signals (Falender & Shafranske / Safran & Muran)
    withdrawal_markers = [
        "随便",
        "都行",
        "你说得对",
        "可能吧",
    ]
    confrontation_markers = [
        "你不理解",
        "你没听懂",
        "不是这样的",
        "你错了",
    ]
    for msg in messages[-3:]:
        content = msg.get("content", "")
        if any(marker in content for marker in withdrawal_markers) and len(content) < 20:
            return "检测到可能的退缩型联盟破裂（简短顺从但可能疏离），建议使用元沟通技术探索关系。"
        if any(marker in content for marker in confrontation_markers):
            return "检测到可能的对抗型联盟破裂（来访者表达不满），建议先承认并探索来访者的不满，不防御。"

    if len(messages) < 2:
        return None

    # Check topic change patterns
    current_topic = context.get("current_topic", "")
    if current_topic and len(messages) >= 4:
        prev_topics = [m.get("content", "")[:30] for m in messages[-4:-2]]
        topic_change = any(current_topic[:20] not in p for p in prev_topics)
        if topic_change:
            return "话题发生转换，注意是否有未完成的探索。"

    return None


async def supervisoner_ainvoke(input: dict) -> Optional[str]:
    """
    Wrapper for supervisor invocation - compatible with top_module call pattern.
    """
    messages = input.get("messages", [])
    ctx = input.get("context", {})

    return await supervisoner(messages, ctx)
