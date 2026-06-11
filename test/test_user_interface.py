# test/test_user_interface.py
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from user_interface import load_command, input_process
from SharedContext import SharedContext


# ---------- Tests for load_command ----------
def test_load_command_non_load():
    """Test load_command with input not starting with /load."""
    is_load, load_id = load_command("hello")
    assert is_load is False
    assert load_id is None


def test_load_command_empty():
    """Test load_command with empty input."""
    is_load, load_id = load_command("")
    assert is_load is False
    assert load_id is None


def test_load_command_only_load():
    """Test load_command with '/load' only."""
    is_load, load_id = load_command("/load")
    assert is_load is True
    assert load_id is None


def test_load_command_with_id():
    """Test load_command with '/load <id>'."""
    is_load, load_id = load_command("/load 123")
    assert is_load is True
    assert load_id == "123"


def test_load_command_with_id_and_spaces():
    """Test load_command with extra spaces."""
    is_load, load_id = load_command("/load   abc")
    assert is_load is True
    assert load_id == "abc"


def test_load_command_with_multiple_parts():
    """Test load_command with multiple parts after /load."""
    is_load, load_id = load_command("/load 123 extra")
    assert is_load is True
    assert load_id == "123"


# ---------- Tests for input_process ----------
@pytest.mark.asyncio
async def test_input_process_exit():
    """Test input_process with /exit command."""
    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = AsyncMock()
    with patch('builtins.input', side_effect=["/exit"]):
        await input_process(mock_ctx)
    # auto_save should have been called
    mock_ctx.auto_save.assert_called_once()


@pytest.mark.asyncio
async def test_input_process_load():
    """Test input_process with /load command."""
    mock_ctx = AsyncMock(spec=SharedContext)
    # Mock the return value of load_from_file
    mock_new_ctx = MagicMock()
    mock_new_ctx._messages = []
    mock_new_ctx._stats = {"total_messages": 0}
    mock_new_ctx._analyst_injection = None
    mock_new_ctx._supervisor_injection = None
    mock_new_ctx._chatter_retrieval_injection = None
    mock_new_ctx.session_id = "123"

    mock_ctx.auto_save = AsyncMock()
    mock_ctx.load_from_file = AsyncMock(return_value=mock_new_ctx)
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx._chatter_retrieval_injection = None
    mock_ctx.analyst_trigger = MagicMock()
    mock_ctx.supervisor_trigger = MagicMock()
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx._messages = []
    mock_ctx._stats = {}
    mock_ctx.session_id = "test"
    mock_ctx.token_limit = 128000
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    mock_ctx.get_token_usage = AsyncMock(return_value={
        "usage_percentage": 50.0, "current_tokens": 1000, "token_limit": 128000,
        "remaining_tokens": 127000, "is_near_limit_80": False,
        "is_near_limit_90": False, "is_over_limit": False,
    })

    with patch('builtins.input', side_effect=["/load 123", "/exit"]):
        with patch('user_interface.call_supervisor') as mock_call_supervisor:
            with patch('user_interface.call_chatter') as mock_call_chatter:
                await input_process(mock_ctx)

    # Verify load_from_file was called with "123"
    mock_ctx.load_from_file.assert_called_once_with("123")
    # call_supervisor should not be called because spare flag is False
    mock_call_supervisor.assert_not_called()
    # call_chatter is not called because load uses continue, skipping message processing
    mock_call_chatter.assert_not_called()


@pytest.mark.asyncio
async def test_input_process_normal_message():
    """Test input_process with a normal user message."""
    from chatter import ChatterOutput

    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = AsyncMock()
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx._chatter_retrieval_injection = None
    mock_ctx.analysist_spare = True
    mock_ctx.supervisor_spare = True
    mock_ctx.get_recent_messages = AsyncMock(return_value=[
        {"role": "user", "content": "previous", "timestamp": 0},
        {"role": "assistant", "content": "prev resp", "timestamp": 1}
    ])
    mock_ctx.add_message = AsyncMock()
    mock_ctx.get_token_usage = AsyncMock(return_value={
        "usage_percentage": 50.0, "current_tokens": 1000, "token_limit": 128000,
        "remaining_tokens": 127000, "is_near_limit_80": False,
        "is_near_limit_90": False, "is_over_limit": False,
    })

    with patch('builtins.input', side_effect=["hello", "/exit"]):
        with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
            with patch('user_interface.call_chatter', new_callable=AsyncMock) as mock_call_chatter:
                mock_call_chatter.return_value = ChatterOutput(
                    reply="mock reply",
                    should_retrieve=False,
                    retrieve_query=""
                )
                await input_process(mock_ctx)

    # call_supervisor should be called because spare flag is True
    mock_call_supervisor.assert_called_once_with(mock_ctx)
    # add_message called for user and assistant
    assert mock_ctx.add_message.call_count == 2
    mock_ctx.add_message.assert_has_calls([
        call("user", "hello"),
        call("assistant", "mock reply")
    ], any_order=False)
    # call_chatter called with context and user input
    mock_call_chatter.assert_called_once()
    call_args = mock_call_chatter.call_args
    assert call_args[0][0] is mock_ctx


@pytest.mark.asyncio
async def test_input_process_with_injections():
    """Test input_process with analyst and supervisor injections."""
    from SharedContext import PromptInjection
    from chatter import ChatterOutput

    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = AsyncMock()
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = PromptInjection(content="analyst note", timestamp=0, source="analyst")
    mock_ctx._supervisor_injection = PromptInjection(content="supervisor note", timestamp=0, source="supervisor")
    mock_ctx._chatter_retrieval_injection = PromptInjection(content="retrieval result", timestamp=0, source="chatter_retrieval")
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx.get_profile_summary = MagicMock(return_value="")
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    mock_ctx.get_token_usage = AsyncMock(return_value={
        "usage_percentage": 50.0, "current_tokens": 1000, "token_limit": 128000,
        "remaining_tokens": 127000, "is_near_limit_80": False,
        "is_near_limit_90": False, "is_over_limit": False,
    })

    with patch('builtins.input', side_effect=["test", "/exit"]):
        with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
            with patch('user_interface.call_chatter', new_callable=AsyncMock) as mock_call_chatter:
                mock_call_chatter.return_value = ChatterOutput(
                    reply="reply",
                    should_retrieve=False,
                    retrieve_query=""
                )
                await input_process(mock_ctx)

    # call_supervisor not called because spare flag False
    mock_call_supervisor.assert_not_called()
    # add_message: injections are prepended in order: supervisor first, then retrieval
    expected_input = "retrieval:retrieval result\nsupervisor:supervisor note\ntest"
    mock_ctx.add_message.assert_has_calls([
        call("user", expected_input),
        call("assistant", "reply")
    ])
    # call_chatter called with the right args
    mock_call_chatter.assert_called_once_with(mock_ctx, expected_input, mock_ctx.get_profile_summary())


@pytest.mark.asyncio
async def test_input_process_no_spare():
    """Test input_process with supervisor_spare False."""
    from chatter import ChatterOutput

    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = AsyncMock()
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx._chatter_retrieval_injection = None
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx.get_profile_summary = MagicMock(return_value="")
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    mock_ctx.get_token_usage = AsyncMock(return_value={
        "usage_percentage": 50.0, "current_tokens": 1000, "token_limit": 128000,
        "remaining_tokens": 127000, "is_near_limit_80": False,
        "is_near_limit_90": False, "is_over_limit": False,
    })

    with patch('builtins.input', side_effect=["msg", "/exit"]):
        with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
            with patch('user_interface.call_chatter', new_callable=AsyncMock) as mock_call_chatter:
                mock_call_chatter.return_value = ChatterOutput(
                    reply="response",
                    should_retrieve=False,
                    retrieve_query=""
                )
                await input_process(mock_ctx)

    mock_call_supervisor.assert_not_called()
    mock_ctx.add_message.assert_has_calls([
        call("user", "msg"),
        call("assistant", "response")
    ])
    mock_call_chatter.assert_called_once_with(mock_ctx, "msg", mock_ctx.get_profile_summary())


if __name__ == "__main__":
    pytest.main([__file__])
