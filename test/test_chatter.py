# test/test_chatter.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatter import call_chatter, ChatterResponse


# ---------- Tests for call_chatter ----------
@pytest.mark.asyncio
async def test_call_chatter_basic():
    """Test basic chatter call."""
    # Create a minimal mock context
    ctx = MagicMock()
    ctx.messages = []
    ctx.analyst_injection = None
    ctx.supervisor_injection = None

    with patch("chatter.base_model") as mock_model:
        mock_response = MagicMock()
        mock_response.reply = "我理解你的感受"
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await call_chatter(ctx)

        assert result == "我理解你的感受"


@pytest.mark.asyncio
async def test_call_chatter_with_analyst_injection():
    """Test chatter incorporates analyst injection."""
    ctx = MagicMock()
    ctx.messages = []
    ctx.analyst_injection = MagicMock(content="焦虑情绪分析")
    ctx.supervisor_injection = None

    with patch("chatter.base_model") as mock_model:
        mock_response = MagicMock()
        mock_response.reply = "我理解你的感受"
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await call_chatter(ctx)

        # Analyst injection should be cleared after use
        assert ctx.analyst_injection is None


@pytest.mark.asyncio
async def test_call_chatter_with_supervisor_injection():
    """Test chatter incorporates supervisor injection."""
    ctx = MagicMock()
    ctx.messages = []
    ctx.analyst_injection = None
    ctx.supervisor_injection = MagicMock(content="温和引导")

    with patch("chatter.base_model") as mock_model:
        mock_response = MagicMock()
        mock_response.reply = "我理解你的感受"
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await call_chatter(ctx)

        # Supervisor injection should be cleared after use
        assert ctx.supervisor_injection is None


@pytest.mark.asyncio
async def test_call_chatter_with_conversation_history():
    """Test chatter uses conversation history."""
    ctx = MagicMock()
    ctx.messages = [
        MagicMock(role="user", content="我最近焦虑"),
        MagicMock(role="assistant", content="嗯，说说看"),
    ]
    ctx.analyst_injection = None
    ctx.supervisor_injection = None

    with patch("chatter.base_model") as mock_model:
        mock_response = MagicMock()
        mock_response.reply = "我理解你的感受"
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await call_chatter(ctx)

        # Verify LLM was called with history
        call_args = mock_model.with_structured_output.return_value.ainvoke.call_args
        messages = call_args[0][0]

        # Should have system message + history messages
        assert len(messages) >= 3


@pytest.mark.asyncio
async def test_call_chatter_injects_both():
    """Test chatter handles both injections."""
    ctx = MagicMock()
    ctx.messages = []
    ctx.analyst_injection = MagicMock(content="分析内容")
    ctx.supervisor_injection = MagicMock(content="指导内容")

    with patch("chatter.base_model") as mock_model:
        mock_response = MagicMock()
        mock_response.reply = "回复"
        mock_model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await call_chatter(ctx)

        # Both injections should be cleared
        assert ctx.analyst_injection is None
        assert ctx.supervisor_injection is None