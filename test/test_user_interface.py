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
    mock_ctx.auto_save = None  # attribute, not callable
    with patch('builtins.input', side_effect=["/exit"]):
        await input_process(mock_ctx)
    # Should return without raising; auto_save attribute accessed
    assert not mock_ctx.method_calls  # no other methods called


@pytest.mark.asyncio
async def test_input_process_load():
    """Test input_process with /load command."""
    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = None
    mock_ctx.load_from_file = AsyncMock()
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    
    with patch('builtins.input', side_effect=["/load 123", "/exit"]):
        with patch('user_interface.call_analysist') as mock_call_analysist:
            with patch('user_interface.call_supervisor') as mock_call_supervisor:
                with patch('user_interface.chatter') as mock_chatter:
                    mock_msg = MagicMock()
                    mock_msg.content = "response"
                    mock_chatter.ainvoke = AsyncMock(return_value={
                        "messages": [mock_msg]
                    })
                    # Run the function, but it will loop; we need to break after first iteration.
                    # Since we patched input to return /load then /exit, the loop will process /load then exit.
                    # Let's use a side effect to stop after first iteration.
                    # Instead, we'll just run and let it exit on /exit.
                    await input_process(mock_ctx)
    
    # Verify load_from_file was called with "123"
    mock_ctx.load_from_file.assert_called_once_with("123")
    # call_analysist and call_supervisor should not be called because spare flags are False
    mock_call_analysist.assert_not_called()
    mock_call_supervisor.assert_not_called()
    # chatter is called even with load command
    mock_chatter.ainvoke.assert_called_once()
    call_args = mock_chatter.ainvoke.call_args
    # Should be called with a dict containing messages
    assert call_args.args[0]["messages"][0]["content"] == "\n\n/load 123"


@pytest.mark.asyncio
async def test_input_process_normal_message():
    """Test input_process with a normal user message."""
    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = None
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx.analysist_spare = True
    mock_ctx.supervisor_spare = True
    mock_ctx.get_recent_messages = AsyncMock(return_value=[
        {"role": "user", "content": "previous", "timestamp": 0},
        {"role": "assistant", "content": "prev resp", "timestamp": 1}
    ])
    mock_ctx.add_message = AsyncMock()
    
    with patch('builtins.input', side_effect=["hello", "/exit"]):
        with patch('user_interface.call_analysist', new_callable=AsyncMock) as mock_call_analysist:
            with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
                with patch('user_interface.chatter') as mock_chatter:
                    mock_msg = MagicMock()
                    mock_msg.content = "mock reply"
                    mock_chatter.ainvoke = AsyncMock(return_value={
                        "messages": [mock_msg]
                    })
                    # We need to stop after first iteration; we'll use a side effect on input to raise StopAsyncIteration
                    # Instead, we can let the loop run twice (first "hello", second "/exit") and exit.
                    await input_process(mock_ctx)
    
    # call_analysist and call_supervisor should be called because spare flags are True
    mock_call_analysist.assert_called_once_with(mock_ctx)
    mock_call_supervisor.assert_called_once_with(mock_ctx)
    # add_message called for user and assistant
    assert mock_ctx.add_message.call_count == 2
    mock_ctx.add_message.assert_has_calls([
        call("user", "hello"),
        call("assistant", "mock reply")
    ], any_order=False)
    # chatter.ainvoke called with appropriate chat_input
    mock_chatter.ainvoke.assert_called_once()
    call_args = mock_chatter.ainvoke.call_args
    # call_args.args[0] is the dict passed to ainvoke
    assert len(call_args.args) == 1
    assert "messages" in call_args.args[0]
    # Ensure chat_input includes previous messages and user input
    chat_input = call_args.args[0]["messages"][0]["content"]
    assert "previous" in chat_input
    assert "prev resp" in chat_input
    assert "hello" in chat_input


@pytest.mark.asyncio
async def test_input_process_with_injections():
    """Test input_process with analyst and supervisor injections."""
    from SharedContext import PromptInjection
    
    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = None
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = PromptInjection(content="analyst note", timestamp=0, source="analyst")
    mock_ctx._supervisor_injection = PromptInjection(content="supervisor note", timestamp=0, source="supervisor")
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    
    with patch('builtins.input', side_effect=["test", "/exit"]):
        with patch('user_interface.call_analysist', new_callable=AsyncMock) as mock_call_analysist:
            with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
                with patch('user_interface.chatter') as mock_chatter:
                    mock_msg = MagicMock()
                    mock_msg.content = "reply"
                    mock_chatter.ainvoke = AsyncMock(return_value={
                        "messages": [mock_msg]
                    })
                    await input_process(mock_ctx)
    
    # call_analysist and call_supervisor not called because spare flags False
    mock_call_analysist.assert_not_called()
    mock_call_supervisor.assert_not_called()
    # add_message should have user input with injections appended
    mock_ctx.add_message.assert_has_calls([
        call("user", "testanlysist:analyst notesupervisor:supervisor note"),
        call("assistant", "reply")
    ])
    # chatter.ainvoke called with injections in chat_input
    call_args = mock_chatter.ainvoke.call_args
    chat_input = call_args.args[0]["messages"][0]["content"]
    assert "anlysist:analyst note" in chat_input
    assert "supervisor:supervisor note" in chat_input


@pytest.mark.asyncio
async def test_input_process_no_spare():
    """Test input_process with analysist_spare and supervisor_spare False."""
    mock_ctx = AsyncMock(spec=SharedContext)
    mock_ctx.auto_save = None
    mock_ctx._lock = AsyncMock()
    mock_ctx._analyst_injection = None
    mock_ctx._supervisor_injection = None
    mock_ctx.analysist_spare = False
    mock_ctx.supervisor_spare = False
    mock_ctx.get_recent_messages = AsyncMock(return_value=[])
    mock_ctx.add_message = AsyncMock()
    
    with patch('builtins.input', side_effect=["msg", "/exit"]):
        with patch('user_interface.call_analysist', new_callable=AsyncMock) as mock_call_analysist:
            with patch('user_interface.call_supervisor', new_callable=AsyncMock) as mock_call_supervisor:
                with patch('user_interface.chatter') as mock_chatter:
                    mock_msg = MagicMock()
                    mock_msg.content = "response"
                    mock_chatter.ainvoke = AsyncMock(return_value={
                        "messages": [mock_msg]
                    })
                    await input_process(mock_ctx)
    
    mock_call_analysist.assert_not_called()
    mock_call_supervisor.assert_not_called()
    mock_ctx.add_message.assert_has_calls([
        call("user", "msg"),
        call("assistant", "response")
    ])
    mock_chatter.ainvoke.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])