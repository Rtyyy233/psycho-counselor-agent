"""
Conversation Manager - Handles long conversation context summarization and storage.

When context approaches the length limit:
1. Generates a summary of the conversation
2. Stores the conversation via store_conv_outline
3. Resets context with summary as new starting point
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
import os
from config import LLM_MODEL
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from top_module import SharedContext, ChatMessage
from mem_store_conv_outline import store_conversation_outline


# ========== Configuration ==========

DEFAULT_MAX_TOKENS = 6000  # Approximate token limit for context
DEFAULT_SUMMARY_THRESHOLD = 0.8  # Trigger summary at 80% of limit
SUMMARY_SYSTEM_PROMPT = """你是一位专业的心理咨询记录分析师。请根据以下对话内容，生成简洁的摘要。

摘要应包含：
- 对话的主要话题和进展
- 用户表达的核心情绪或问题
- 咨询师的关键回应方向
- 双方达成的一致或待解决的问题

保持简洁，200字以内。"""


class ConversationSummary(BaseModel):
    """Summary of a conversation segment."""
    main_topic: str = Field(description="主要话题")
    key_emotions: list[str] = Field(default_factory=list, description="核心情绪")
    progress_summary: str = Field(default="", description="进展摘要")
    unresolved_issues: list[str] = Field(default_factory=list, description="待解决问题")
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationSegment:
    """A segment of conversation for storage."""
    messages: list[ChatMessage]
    summary: Optional[ConversationSummary] = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None


# ========== Conversation Manager ==========

class ConversationManager:
    """
    Manages conversation context with automatic summarization.

    When context length approaches limit:
    1. Summarizes current conversation segment
    2. Stores via store_conv_outline
    3. Resets context with summary
    """

    def __init__(
        self,
        ctx: SharedContext,
        llm: Optional[ChatDeepSeek] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        threshold: float = DEFAULT_SUMMARY_THRESHOLD
    ):
        self.ctx = ctx
        self.llm = llm or ChatDeepSeek(model=LLM_MODEL, temperature=0.3)
        self.max_tokens = max_tokens
        self.threshold = threshold

        # Track conversation segments
        self.segments: list[ConversationSegment] = []
        self.current_segment: Optional[ConversationSegment] = None
        self._total_messages_processed = 0

    def estimate_tokens(self, messages: list[ChatMessage]) -> int:
        """Estimate token count from messages."""
        # Rough estimate: ~4 chars per token for Chinese
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4 + len(messages) * 2  # Account for role overhead

    def should_summarize(self) -> bool:
        """Check if context length is approaching limit."""
        current_tokens = self.estimate_tokens(self.ctx.messages)
        return current_tokens >= self.max_tokens * self.threshold

    async def generate_summary(self, messages: list[ChatMessage]) -> ConversationSummary:
        """Generate a summary of the conversation segment."""
        if not messages:
            return ConversationSummary(main_topic="(空对话)")

        # Format conversation for LLM
        formatted = "\n".join([
            f"{'用户' if m.role == 'user' else '咨询师'}: {m.content}"
            for m in messages[-20:]  # Last 20 messages for summary
        ])

        prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\n对话内容：\n{formatted}"

        structured_llm = self.llm.with_structured_output(ConversationSummary)

        try:
            summary = await structured_llm.ainvoke(prompt)
            return summary
        except Exception as e:
            # Fallback: simple summary
            return ConversationSummary(
                main_topic=messages[-1].content[:50] if messages else "(空)",
                progress_summary=f"共 {len(messages)} 条消息"
            )

    async def store_segment(self, messages: list[ChatMessage], summary: ConversationSummary) -> str:
        """Store conversation segment via store_conv_outline."""
        # Format as document
        full_text = "\n".join([
            f"[{'用户' if m.role == 'user' else '咨询师'}] {m.content}"
            for m in messages
        ])

        # Build PAIP-style structured content for storage
        paip_content = f"""【问题】
{summary.main_topic}

【情绪】
{', '.join(summary.key_emotions) if summary.key_emotions else '未明确'}

【进展】
{summary.progress_summary}

【待解决】
{', '.join(summary.unresolved_issues) if summary.unresolved_issues else '无'}"""

        doc = Document(
            page_content=full_text,
            metadata={
                "source": "conversation_manager",
                "summary": summary.progress_summary,
                "message_count": len(messages)
            }
        )

        try:
            result = await store_conversation_outline(doc)
            return result
        except Exception as e:
            return f"存储失败: {str(e)}"

    async def reset_context_with_summary(self, summary: ConversationSummary):
        """Reset context with summary as new starting point."""
        # Clear current messages but keep structure
        async with self.ctx._lock:
            self.ctx.messages.clear()
            self.ctx.topic_history.clear()
            self.ctx.current_topic = summary.main_topic

        # Add summary as a system message
        summary_text = (
            f"【对话摘要 - {summary.timestamp.strftime('%Y-%m-%d %H:%M')}】\n"
            f"话题：{summary.main_topic}\n"
            f"情绪：{', '.join(summary.key_emotions) if summary.key_emotions else '未明确'}\n"
            f"进展：{summary.progress_summary}\n"
            f"待解决：{', '.join(summary.unresolved_issues) if summary.unresolved_issues else '无'}"
        )

        await self.ctx.add_message("system", summary_text)

    async def manage(self) -> Optional[str]:
        """
        Main management function. Call this periodically or after each exchange.

        Returns:
            str: "summarized" if summary was triggered, None otherwise
        """
        if not self.should_summarize():
            return None

        # Get current messages for this segment
        segment_messages = list(self.ctx.messages)

        # Generate summary
        summary = await self.generate_summary(segment_messages)

        # Store segment
        store_result = await self.store_segment(segment_messages, summary)

        # Reset context with summary
        await self.reset_context_with_summary(summary)

        # Track segment
        segment = ConversationSegment(
            messages=segment_messages,
            summary=summary,
            ended_at=datetime.now()
        )
        self.segments.append(segment)

        self._total_messages_processed += len(segment_messages)

        return store_result

    def get_segment_count(self) -> int:
        """Return number of stored segments."""
        return len(self.segments)

    def get_total_messages_processed(self) -> int:
        """Return total messages across all segments."""
        return self._total_messages_processed


# ========== Async Background Manager ==========

async def run_conversation_manager(ctx: SharedContext, llm: ChatDeepSeek):
    """
    Background task that monitors context and triggers summarization.

    Uses an event-based approach:
    - Monitors ctx.on_new_message
    - Checks context length after each message
    - Triggers summarization when threshold is reached
    """
    manager = ConversationManager(ctx, llm)

    while True:
        # Wait for new message
        await ctx.on_new_message.wait()
        ctx.on_new_message.clear()

        # Check if summarization is needed
        result = await manager.manage()

        if result:
            # Inject notification about summarization
            await ctx.safe_set_supervisor(
                f"对话已总结并存储（共{manager.get_segment_count()}段）。{result}",
                priority="gentle"
            )

        await asyncio.sleep(0.1)


# ========== Standalone Utility Functions ==========

async def summarize_now(ctx: SharedContext) -> Optional[ConversationSummary]:
    """Manual trigger for summarization."""
    manager = ConversationManager(ctx)
    if len(ctx.messages) < 2:
        return None

    summary = await manager.generate_summary(ctx.messages)
    await manager.store_segment(list(ctx.messages), summary)
    await manager.reset_context_with_summary(summary)

    return summary


async def get_context_stats(ctx: SharedContext) -> dict:
    """Get current context statistics."""
    manager = ConversationManager(ctx)
    return {
        "message_count": len(ctx.messages),
        "estimated_tokens": manager.estimate_tokens(ctx.messages),
        "max_tokens": manager.max_tokens,
        "usage_percent": manager.estimate_tokens(ctx.messages) / manager.max_tokens * 100,
        "segments_stored": manager.get_segment_count()
    }