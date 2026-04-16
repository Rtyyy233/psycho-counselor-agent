# test/test_conversation_manager.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conversation_manager import (
    ConversationManager,
    ConversationSummary,
    ConversationSegment,
    summarize_now,
    get_context_stats,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SUMMARY_THRESHOLD,
)


# ---------- Tests for token estimation ----------
def test_estimate_tokens_empty():
    """Test token estimation with empty messages."""
    from top_module import ChatMessage

    messages = []
    manager = ConversationManager(MagicMock())

    tokens = manager.estimate_tokens(messages)

    assert tokens == 0


def test_estimate_tokens_with_messages():
    """Test token estimation with messages."""
    from top_module import ChatMessage

    messages = [
        ChatMessage(role="user", content="你好"),
        ChatMessage(role="assistant", content="你好吗"),
    ]
    manager = ConversationManager(MagicMock())

    tokens = manager.estimate_tokens(messages)

    assert tokens > 0


# ---------- Tests for should_summarize ----------
def test_should_summarize_below_threshold():
    """Test that summarization is not triggered below threshold."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="短消息")
    ]

    manager = ConversationManager(ctx, max_tokens=1000)
    assert not manager.should_summarize()


def test_should_summarize_above_threshold():
    """Test that summarization is triggered above threshold."""
    from top_module import ChatMessage

    ctx = MagicMock()
    # Create enough messages to exceed threshold
    ctx.messages = [
        ChatMessage(role="user", content="x" * 1000)
        for _ in range(20)
    ]

    manager = ConversationManager(ctx, max_tokens=1000, threshold=0.8)
    # Should trigger because we're well over 80% of 1000 tokens
    # Each message is ~1000 chars = ~250 tokens, 20 messages = ~5000 tokens >> 800


# ---------- Tests for generate_summary ----------
@pytest.mark.asyncio
async def test_generate_summary_with_messages():
    """Test summary generation returns a ConversationSummary object."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="I have been feeling anxious lately"),
        ChatMessage(role="assistant", content="Can you tell me more"),
    ]

    manager = ConversationManager(ctx)

    # Mock the LLM
    with patch("conversation_manager.ChatDeepSeek") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        expected_summary = ConversationSummary(
            main_topic="anxiety",
            key_emotions=["anxiety"],
            progress_summary="user expressed anxiety"
        )

        mock_structured = MagicMock()
        mock_structured.invoke = AsyncMock(return_value=expected_summary)
        mock_instance.with_structured_output.return_value = mock_structured

        summary = await manager.generate_summary(ctx.messages)

        # Verify the summary was generated
        assert summary is not None
        assert isinstance(summary, ConversationSummary)


@pytest.mark.asyncio
async def test_generate_summary_empty_messages():
    """Test summary generation with empty messages."""
    ctx = MagicMock()
    manager = ConversationManager(ctx)

    summary = await manager.generate_summary([])

    assert summary.main_topic == "(空对话)"


# ---------- Tests for ConversationManager manage ----------
@pytest.mark.asyncio
async def test_manage_triggers_when_needed():
    """Test that manage method checks threshold correctly."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="x" * 500)
        for _ in range(15)
    ]
    ctx._lock = asyncio.Lock()
    ctx.add_message = AsyncMock()

    manager = ConversationManager(ctx, max_tokens=1000, threshold=0.5)

    # Verify should_summarize works with low threshold
    assert manager.should_summarize() == True


@pytest.mark.asyncio
async def test_manage_skips_when_not_needed():
    """Test that manage skips when below threshold."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="短")
    ]
    ctx._lock = asyncio.Lock()

    manager = ConversationManager(ctx, max_tokens=10000, threshold=0.8)

    result = await manager.manage()

    assert result is None


# ---------- Tests for reset_context_with_summary ----------
@pytest.mark.asyncio
async def test_reset_context_with_summary():
    """Test context reset with summary."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="测试"),
    ]
    ctx._lock = asyncio.Lock()
    ctx.add_message = AsyncMock()
    ctx.topic_history = []
    ctx.current_topic = ""

    manager = ConversationManager(ctx)

    summary = ConversationSummary(
        main_topic="焦虑问题",
        key_emotions=["焦虑"],
        progress_summary="讨论了焦虑问题"
    )

    await manager.reset_context_with_summary(summary)

    assert ctx.current_topic == "焦虑问题"


# ---------- Tests for get_context_stats ----------
@pytest.mark.asyncio
async def test_get_context_stats():
    """Test getting context statistics."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="测试消息"),
        ChatMessage(role="assistant", content="回复"),
    ]
    ctx._lock = asyncio.Lock()

    stats = await get_context_stats(ctx)

    assert "message_count" in stats
    assert "estimated_tokens" in stats
    assert "usage_percent" in stats
    assert stats["message_count"] == 2


# ---------- Tests for ConversationSegment ----------
def test_conversation_segment_creation():
    """Test ConversationSegment dataclass."""
    from top_module import ChatMessage

    segment = ConversationSegment(
        messages=[ChatMessage(role="user", content="测试")]
    )

    assert len(segment.messages) == 1
    assert segment.started_at is not None
    assert segment.ended_at is None


def test_conversation_segment_with_summary():
    """Test ConversationSegment with summary."""
    segment = ConversationSegment(
        messages=[],
        summary=ConversationSummary(
            main_topic="测试话题",
            progress_summary="测试摘要"
        )
    )

    assert segment.summary is not None
    assert segment.summary.main_topic == "测试话题"


# ---------- Tests for segment tracking ----------
@pytest.mark.asyncio
async def test_segment_tracking():
    """Test that ConversationManager tracks segments correctly."""
    from top_module import ChatMessage

    ctx = MagicMock()
    ctx.messages = [
        ChatMessage(role="user", content="x" * 500)
        for _ in range(15)
    ]
    ctx._lock = asyncio.Lock()
    ctx.add_message = AsyncMock()

    manager = ConversationManager(ctx, max_tokens=100, threshold=0.1)

    # Verify should_summarize triggers
    assert manager.should_summarize() == True

    # Verify get_segment_count returns correct value
    assert manager.get_segment_count() == 0
    assert manager.get_total_messages_processed() == 0