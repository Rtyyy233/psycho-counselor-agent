# test/test_analysist.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysist import call_analysist, synthesize_analysis


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
async def test_call_analysist_calls_retrieval_modules():
    """Test analyst calls retrieval modules."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[
            MagicMock(role="user", content="我最近焦虑"),
            MagicMock(role="assistant", content="嗯"),
        ]
    )

    with (
        patch("analysist.retrieve_diary", new_callable=AsyncMock) as mock_diary,
        patch("analysist.retrieve_materials", new_callable=AsyncMock) as mock_mat,
        patch("analysist.retrieve_conv_outline", new_callable=AsyncMock) as mock_conv,
    ):
        mock_diary.return_value = []
        mock_mat.return_value = []
        mock_conv.return_value = []

        result = await call_analysist(ctx)

        mock_diary.assert_called_once()
        mock_mat.assert_called_once()
        mock_conv.assert_called_once()


@pytest.mark.asyncio
async def test_call_analysist_returns_analysis():
    """Test analyst returns synthesized analysis."""
    ctx = MagicMock()
    ctx.get_recent_messages = AsyncMock(
        return_value=[
            MagicMock(role="user", content="我最近焦虑"),
            MagicMock(role="assistant", content="嗯"),
        ]
    )

    with (
        patch("analysist.retrieve_diary", new_callable=AsyncMock) as mock_diary,
        patch("analysist.retrieve_materials", new_callable=AsyncMock) as mock_mat,
        patch("analysist.retrieve_conv_outline", new_callable=AsyncMock) as mock_conv,
    ):
        mock_diary.return_value = [{"id": "1"}]  # Some results
        mock_mat.return_value = []
        mock_conv.return_value = []

        result = await call_analysist(ctx)

        assert result is not None
        assert "日记" in result or "焦虑" in result


# ---------- Tests for synthesize_analysis ----------
def test_synthesize_analysis_no_results():
    """Test synthesis with empty results."""
    result = synthesize_analysis(
        query="test query", diary=[], materials=[], conv_outline=[]
    )

    assert "暂无相关记忆" in result


def test_synthesize_analysis_with_diary():
    """Test synthesis with diary results."""
    result = synthesize_analysis(
        query="焦虑", diary=[{"id": "1"}, {"id": "2"}], materials=[], conv_outline=[]
    )

    assert "日记" in result
    assert "2" in result


def test_synthesize_analysis_with_materials():
    """Test synthesis with material results."""
    result = synthesize_analysis(
        query="人生感悟", diary=[], materials=[{"id": "1"}], conv_outline=[]
    )

    assert "个人材料" in result


def test_synthesize_analysis_with_conv():
    """Test synthesis with conversation outline results."""
    result = synthesize_analysis(
        query="咨询", diary=[], materials=[], conv_outline=[{"id": "1"}]
    )

    assert "咨询记录" in result


def test_synthesize_analysis_with_all():
    """Test synthesis with all result types."""
    result = synthesize_analysis(
        query="焦虑",
        diary=[{"id": "1"}],
        materials=[{"id": "1"}, {"id": "2"}],
        conv_outline=[{"id": "1"}],
    )

    assert "日记" in result
    assert "个人材料" in result
    assert "咨询记录" in result
