# test/test_analysist.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysist import call_analysist, AnalystResult


# ---------- Tests for call_analysist ----------
@pytest.mark.asyncio
async def test_call_analysist_returns_none_with_insufficient_messages():
    """Test analyst returns None when not enough messages."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[MagicMock(role="user", content="hi")]
    )

    result = await call_analysist(ctx)

    assert result is None


@pytest.mark.asyncio
async def test_call_analysist_returns_analysis():
    """Test analyst returns analysis when agent provides valid response."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[
            MagicMock(role="user", content="我最近焦虑"),
            MagicMock(role="assistant", content="嗯"),
        ]
    )

    # Mock the agent to return a valid response
    mock_agent_response = MagicMock()
    mock_agent_response.content = (
        "分析：用户表达了焦虑情绪。建议探索焦虑的来源和应对策略。"
    )

    with patch("analysist.analyst_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(return_value=mock_agent_response)

        result = await call_analysist(ctx)

        assert result is not None
        assert "焦虑" in result or "分析" in result
        mock_agent.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_call_analysist_returns_none_with_empty_analysis():
    """Test analyst returns None when analysis is too short."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[
            MagicMock(role="user", content="我最近焦虑"),
            MagicMock(role="assistant", content="嗯"),
        ]
    )

    # Mock the agent to return a short/empty response
    mock_agent_response = MagicMock()
    mock_agent_response.content = ""

    with patch("analysist.analyst_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(return_value=mock_agent_response)

        result = await call_analysist(ctx)

        assert result is None


@pytest.mark.asyncio
async def test_call_analysist_handles_exception():
    """Test analyst handles exceptions gracefully."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[
            MagicMock(role="user", content="我最近焦虑"),
            MagicMock(role="assistant", content="嗯"),
        ]
    )

    with patch("analysist.analyst_agent") as mock_agent:
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("Test error"))

        result = await call_analysist(ctx)

        assert result is not None
        assert "错误" in result or "Test error" in result


# ---------- Tests for AnalystResult model ----------
def test_analyst_result_model():
    """Test AnalystResult Pydantic model."""
    result = AnalystResult(
        insight="用户表现出焦虑模式",
        direction="建议探索焦虑触发因素",
        relevant_memories=["日记记录1", "材料记录2"],
    )

    assert result.insight == "用户表现出焦虑模式"
    assert result.direction == "建议探索焦虑触发因素"
    assert len(result.relevant_memories) == 2
    assert "日记记录1" in result.relevant_memories
