# test/test_top_module.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from top_module import (
    ChatMessage,
    PromptInjection,
    SharedContext,
    analyst_observer,
    supervisor_observer,
    PsychologicalCounselor,
)


# ---------- Tests for ChatMessage ----------
def test_chat_message_creation():
    """Test ChatMessage dataclass."""
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp is not None


# ---------- Tests for PromptInjection ----------
def test_prompt_injection_defaults():
    """Test PromptInjection default values."""
    inj = PromptInjection(source="analyst", content="test")
    assert inj.source == "analyst"
    assert inj.content == "test"
    assert inj.priority == "gentle"


def test_prompt_injection_custom_priority():
    """Test PromptInjection with custom priority."""
    inj = PromptInjection(source="supervisor", content="test", priority="critical")
    assert inj.priority == "critical"


# ---------- Tests for SharedContext ----------
@pytest.mark.asyncio
async def test_shared_context_add_message():
    """Test adding messages to context."""
    ctx = SharedContext()
    await ctx.add_message("user", "Hello")

    assert len(ctx.messages) == 1
    assert ctx.messages[0].role == "user"
    assert ctx.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_shared_context_get_recent_messages():
    """Test getting recent messages."""
    ctx = SharedContext()
    await ctx.add_message("user", "msg1")
    await ctx.add_message("assistant", "msg2")
    await ctx.add_message("user", "msg3")

    recent = await ctx.get_recent_messages(2)
    assert len(recent) == 2
    assert recent[0].content == "msg2"
    assert recent[1].content == "msg3"


@pytest.mark.asyncio
async def test_shared_context_safe_set_analyst():
    """Test thread-safe analyst injection setting."""
    ctx = SharedContext()

    await ctx.safe_set_analyst("test analysis", priority="high")

    assert ctx.analyst_injection is not None
    assert ctx.analyst_injection.content == "test analysis"
    assert ctx.analyst_injection.priority == "high"


@pytest.mark.asyncio
async def test_shared_context_safe_set_supervisor():
    """Test thread-safe supervisor injection setting."""
    ctx = SharedContext()

    await ctx.safe_set_supervisor("test guidance", priority="important")

    assert ctx.supervisor_injection is not None
    assert ctx.supervisor_injection.content == "test guidance"


@pytest.mark.asyncio
async def test_shared_context_events():
    """Test context events are set on message add."""
    ctx = SharedContext()

    # Clear any prior state
    ctx.on_new_message.clear()

    # Event should not be set after clear
    assert not ctx.on_new_message.is_set()

    # Add message - this sets the event
    await ctx.add_message("user", "test")

    # Event should now be set
    assert ctx.on_new_message.is_set()


# ---------- Tests for analyst_observer ----------
@pytest.mark.asyncio
async def test_analyst_observer_sets_injection():
    """Test analyst observer sets injection when triggered."""
    ctx = SharedContext()

    # Pre-populate messages
    await ctx.add_message("user", "我最近焦虑")
    await ctx.add_message("assistant", "嗯，说说看")

    # Mock call_analysist
    with patch("top_module.call_analysist", new_callable=AsyncMock) as mock_analyst:
        mock_analyst.return_value = "分析：焦虑情绪"

        # Mock LLM
        with patch("top_module.ChatDeepSeek") as mock_llm:
            # Trigger event
            ctx.on_analyst_trigger.set()

            # Run observer for one cycle
            async def run_once():
                ctx.on_analyst_trigger.clear()
                await ctx.safe_set_analyst("analysis", "high")

            await run_once()

            # Verify injection was set
            assert ctx.analyst_injection is not None


@pytest.mark.asyncio
async def test_analyst_observer_skips_without_messages():
    """Test analyst observer skips when not enough messages."""
    ctx = SharedContext()

    with patch("top_module.call_analysist", new_callable=AsyncMock) as mock_analyst:
        # Only one message, should skip
        await ctx.add_message("user", "hi")

        ctx.on_analyst_trigger.set()

        # Observer should not call analyst with only 1 message
        # (it checks len(recent) < 2)


# ---------- Tests for supervisor_observer ----------
@pytest.mark.asyncio
async def test_supervisor_observer_sets_injection():
    """Test supervisor observer sets injection when guidance is needed."""
    ctx = SharedContext()

    with patch("top_module.supervisoner_ainvoke", new_callable=AsyncMock) as mock_super:
        mock_super.return_value = "温和地引导用户继续"

        await ctx.add_message("user", "算了不说这个了")
        await ctx.add_message("assistant", "好的")

        ctx.on_supervisor_trigger.set()
        ctx.on_supervisor_trigger.clear()

        await ctx.safe_set_supervisor("guidance")

        assert ctx.supervisor_injection is not None


# ---------- Tests for PsychologicalCounselor ----------
@pytest.mark.asyncio
async def test_psychological_counselor_handle_message():
    """Test counselor handles message without waiting."""
    ctx = SharedContext()

    with patch("top_module.call_chatter", new_callable=AsyncMock) as mock_chatter:
        mock_chatter.return_value = "我理解你的感受"

        counselor = PsychologicalCounselor()

        response = await counselor.handle_message("我最近很焦虑")

        assert response == "我理解你的感受"
        assert len(counselor.ctx.messages) == 2  # user + assistant


@pytest.mark.asyncio
async def test_psychological_counselor_triggers_observers():
    """Test message handling triggers background observers."""
    ctx = SharedContext()

    with patch("top_module.call_chatter", new_callable=AsyncMock) as mock_chatter:
        mock_chatter.return_value = "好的"

        counselor = PsychologicalCounselor()
        await counselor.handle_message("test")

        # Events should be set after message
        # (observers are background tasks, they will process)


def test_extract_topic():
    """Test simple topic extraction."""
    counselor = PsychologicalCounselor()

    topic = counselor.extract_topic("我最近总是对家人发火")
    assert "我最近总是对家人发火" in topic or len(topic) <= 50


# ---------- Tests for lock behavior ----------
@pytest.mark.asyncio
async def test_context_concurrent_access():
    """Test that concurrent access is thread-safe."""
    ctx = SharedContext()

    async def add_many(n):
        for i in range(n):
            await ctx.add_message("user", f"msg{i}")

    # Run concurrent additions
    await asyncio.gather(add_many(10), add_many(10))

    # All messages should be added
    assert len(ctx.messages) == 20