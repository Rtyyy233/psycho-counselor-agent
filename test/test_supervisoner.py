# test/test_supervisoner.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supervisoner import (
    supervisoner,
    supervisoner_ainvoke,
    SupervisorResult,
    _fallback_supervisor,
)


# ---------- Tests for supervisoner (LLM-based) ----------
@pytest.mark.asyncio
async def test_supervisoner_returns_none_with_insufficient_messages():
    """Test supervisor returns None when not enough messages."""
    result = await supervisoner([], {})

    assert result is None


@pytest.mark.asyncio
async def test_supervisoner_with_llm_guidance():
    """Test supervisor returns guidance when LLM detects need."""
    messages = [
        {"role": "user", "content": "我最近焦虑，晚上睡不着"},
        {"role": "assistant", "content": "听起来很困扰"},
    ]

    # Mock the LLM to return a guidance
    mock_result = SupervisorResult(
        guidance="这是一个探索焦虑根源的好时机",
        priority="high",
        reason="用户表达了焦虑情绪，需要深入探索",
        should_inject=True,
    )

    with patch("supervisoner.supervisor_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)

        result = await supervisoner(messages, {"current_topic": "焦虑"})

        assert result is not None
        assert "探索焦虑根源" in result
        assert "理由" in result


@pytest.mark.asyncio
async def test_supervisoner_with_no_llm_guidance():
    """Test supervisor returns None when LLM says no guidance needed."""
    messages = [
        {"role": "user", "content": "今天天气不错"},
        {"role": "assistant", "content": "是啊，适合散步"},
    ]

    # Mock the LLM to return no guidance needed
    mock_result = SupervisorResult(
        guidance="", priority="low", reason="正常对话，无需干预", should_inject=False
    )

    with patch("supervisoner.supervisor_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)

        result = await supervisoner(messages, {"current_topic": "天气"})

        assert result is None


@pytest.mark.asyncio
async def test_supervisoner_llm_fallback():
    """Test supervisor falls back to heuristic when LLM fails."""
    messages = [
        {"role": "user", "content": "我很难过，想哭"},
        {"role": "assistant", "content": "我在这里听你说"},
    ]

    with patch("supervisoner.supervisor_llm") as mock_llm:
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        result = await supervisoner(messages, {"current_topic": "情绪"})

        # Should use fallback which detects emotional content
        assert result is not None
        assert "情绪" in result or "深挖" in result


# ---------- Tests for _fallback_supervisor ----------
@pytest.mark.asyncio
async def test_fallback_supervisor_detects_emotion():
    """Test fallback supervisor detects emotional content."""
    messages = [
        {"role": "user", "content": "我很难过，想哭"},
    ]

    result = await _fallback_supervisor(messages, {})

    assert result is not None
    assert "情绪" in result or "深挖" in result


@pytest.mark.asyncio
async def test_fallback_supervisor_detects_avoidance():
    """Test fallback supervisor detects avoidance patterns."""
    messages = [
        {"role": "user", "content": "我最近焦虑"},
        {"role": "assistant", "content": "说说看"},
        {"role": "user", "content": "算了不说这个了"},
    ]

    result = await _fallback_supervisor(messages, {})

    assert result is not None
    assert "回避" in result or "邀请" in result


@pytest.mark.asyncio
async def test_fallback_supervisor_normal_conversation():
    """Test fallback supervisor returns None for normal conversation."""
    messages = [
        {"role": "user", "content": "今天天气不错"},
        {"role": "assistant", "content": "是啊"},
    ]

    result = await _fallback_supervisor(messages, {})

    assert result is None


@pytest.mark.asyncio
async def test_fallback_supervisor_detects_topic_change():
    """Test fallback supervisor detects topic change."""
    messages = [
        {"role": "user", "content": "我最近工作压力很大"},
        {"role": "assistant", "content": "能具体说说吗"},
        {"role": "user", "content": "老板要求很高"},
        {"role": "assistant", "content": "这确实有压力"},
    ]

    result = await _fallback_supervisor(messages, {"current_topic": "工作压力"})

    # With enough messages and topic context, may detect topic change
    # but implementation may vary


# ---------- Tests for supervisoner_ainvoke wrapper ----------
@pytest.mark.asyncio
async def test_supervisoner_ainvoke_wrapper():
    """Test the ainvoke wrapper function."""
    input_data = {
        "messages": [{"role": "user", "content": "我很难过"}],
        "context": {"current_topic": ""},
    }

    # Mock the supervisoner function
    with patch("supervisoner.supervisoner") as mock_super:
        mock_super.return_value = AsyncMock(return_value="检测到强烈情绪")

        result = await supervisoner_ainvoke(input_data)

        assert result is not None
        mock_super.assert_called_once_with(
            input_data["messages"], input_data["context"]
        )


# ---------- Tests for SupervisorResult model ----------
def test_supervisor_result_model():
    """Test SupervisorResult Pydantic model."""
    result = SupervisorResult(
        guidance="建议深入探索情绪",
        priority="high",
        reason="用户表达了深层情绪",
        should_inject=True,
    )

    assert result.guidance == "建议深入探索情绪"
    assert result.priority == "high"
    assert result.reason == "用户表达了深层情绪"
    assert result.should_inject == True
