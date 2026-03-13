"""Tests for wall-clock timeout in spawn_cli and disconnect cleanup in app."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_openai_proxy.claude_cli import spawn_cli

# ── spawn_cli total wall-clock timeout ──────────────────────────────────────


@pytest.mark.asyncio
async def test_spawn_cli_total_timeout_kills_process():
    """When max_request_timeout is exceeded the generator stops and the
    process is killed."""
    lines_yielded: list[str] = []

    async def slow_readline():
        await asyncio.sleep(0.3)
        return (
            b'{"type":"assistant","message":{"content":'
            b'[{"type":"text","text":"hi"}]}}\n'
        )

    mock_stdout = MagicMock()
    mock_stdout.readline = slow_readline

    mock_stderr = AsyncMock()
    mock_stderr.read = AsyncMock(return_value=b"")

    mock_proc = AsyncMock()
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = mock_stderr
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch(
        "claude_openai_proxy.claude_cli.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        gen = spawn_cli(
            prompt="test",
            system_prompt="sys",
            model="sonnet",
            timeout=300,
            max_request_timeout=1,
        )
        async for line in gen:
            lines_yielded.append(line)

    assert len(lines_yielded) <= 4
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_spawn_cli_per_line_timeout():
    """When a single readline takes longer than per-line timeout, the
    generator stops."""

    async def hang_readline():
        await asyncio.sleep(10)
        return b"never\n"

    mock_stdout = MagicMock()
    mock_stdout.readline = hang_readline

    mock_stderr = AsyncMock()
    mock_stderr.read = AsyncMock(return_value=b"")

    mock_proc = AsyncMock()
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = mock_stderr
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch(
        "claude_openai_proxy.claude_cli.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        gen = spawn_cli(
            prompt="test",
            system_prompt="sys",
            model="sonnet",
            timeout=1,
            max_request_timeout=60,
        )
        lines = [line async for line in gen]

    assert lines == []
    mock_proc.kill.assert_called_once()


# ── Streaming disconnect ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_streaming_with_disconnect_stops_on_disconnect():
    from claude_openai_proxy.app import _streaming_with_disconnect

    call_count = 0

    async def fake_inner():
        nonlocal call_count
        for i in range(10):
            call_count += 1
            yield f"data: chunk{i}\n\n"

    mock_request = AsyncMock()
    mock_request.is_disconnected = AsyncMock(side_effect=[False, False, True])

    chunks = []
    async for chunk in _streaming_with_disconnect(fake_inner(), mock_request):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert call_count == 3


@pytest.mark.asyncio
async def test_streaming_with_disconnect_passes_all_when_connected():
    from claude_openai_proxy.app import _streaming_with_disconnect

    async def fake_inner():
        for i in range(3):
            yield f"data: chunk{i}\n\n"

    mock_request = AsyncMock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    chunks = []
    async for chunk in _streaming_with_disconnect(fake_inner(), mock_request):
        chunks.append(chunk)

    assert len(chunks) == 3


# ── Non-streaming disconnect ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_with_disconnect_returns_result_when_connected():
    from claude_openai_proxy.app import _complete_with_disconnect

    sentinel = {"id": "test", "choices": []}

    async def fake_lines():
        yield '{"type":"result"}'

    mock_request = AsyncMock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    with patch(
        "claude_openai_proxy.app.build_complete_response", return_value=sentinel
    ):
        result = await _complete_with_disconnect(fake_lines(), "model", mock_request)

    assert result is sentinel


@pytest.mark.asyncio
async def test_complete_with_disconnect_cancels_on_disconnect():
    from claude_openai_proxy.app import _complete_with_disconnect

    async def slow_lines():
        await asyncio.sleep(10)
        yield "never"

    async def slow_build(lines, model):
        async for _ in lines:
            pass
        return {}

    mock_request = AsyncMock()
    mock_request.is_disconnected = AsyncMock(side_effect=[False, True])

    with (
        patch(
            "claude_openai_proxy.app.build_complete_response", side_effect=slow_build
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await _complete_with_disconnect(slow_lines(), "model", mock_request)
