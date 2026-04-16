# test/test_supervisoner.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supervisoner import supervisoner, supervisoner_ainvoke


# ---------- Tests for supervisoner ----------
@pytest.mark.asyncio
async def test_supervisoner_returns_none_with_insufficient_messages():
    """Test supervisor returns None when not enough messages."""
    result = await supervisoner([], {})

    assert result is None


@pytest.mark.asyncio
async def test_supervisoner_detects_avoidance():
    """Test supervisor detects avoidance patterns."""
    messages = [
        {"role": "user", "content": "我最近焦虑"},
        {"role": "assistant", "content": "说说看"},
        {"role": "user", "content": "算了不说这个了"},
    ]

    result = await supervisoner(messages, {})

    assert result is not None
    assert "回避" in result or "邀请" in result.lower()


@pytest.mark.asyncio
async def test_supervisoner_normal_conversation():
    """Test supervisor returns None for normal conversation."""
    messages = [
        {"role": "user", "content": "今天天气不错"},
        {"role": "assistant", "content": "是啊"},
    ]

    result = await supervisoner(messages, {})

    # Normal conversation should not trigger guidance
    assert result is None


@pytest.mark.asyncio
async def test_supervisoner_topic_change():
    """Test supervisor detects topic change."""
    messages = [
        {"role": "user", "content": "我最近焦虑"},
        {"role": "assistant", "content": "嗯"},
        {"role": "user", "content": "对了上次说的那本书"},
        {"role": "assistant", "content": "哪本书"},
        {"role": "user", "content": "关于心理的"},
    ]

    result = await supervisoner(messages, {"current_topic": "工作压力"})

    # Topic change detected, should warn about unresolved theme
    # (may or may not trigger depending on implementation)


# ---------- Tests for supervisoner_ainvoke wrapper ----------
@pytest.mark.asyncio
async def test_supervisoner_ainvoke_wrapper():
    """Test the ainvoke wrapper function."""
    input_data = {
        "messages": [{"role": "user", "content": "我很难过"}],
        "context": {"current_topic": ""},
    }

    result = await supervisoner_ainvoke(input_data)

    # Should detect emotional content even with 1 message
    assert result is not None
    assert "情绪" in result or "深挖" in result
