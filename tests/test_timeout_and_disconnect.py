"""Tests for streaming disconnect cleanup in the app."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

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


# ── API timeout handling ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_timeout_returns_504():
    """When the Anthropic SDK raises a timeout, the endpoint returns 504."""
    from unittest.mock import patch

    from anthropic import APITimeoutError

    from claude_openai_proxy.app import chat_completions
    from claude_openai_proxy.models import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    mock_raw_request = AsyncMock()
    mock_raw_request.is_disconnected = AsyncMock(return_value=False)

    with patch(
        "claude_openai_proxy.app.create_message",
        side_effect=APITimeoutError(request=None),
    ):
        response = await chat_completions(request, mock_raw_request)

    assert response.status_code == 504


@pytest.mark.asyncio
async def test_anthropic_status_error_forwards_code():
    """When the Anthropic SDK raises APIStatusError, its status code is forwarded."""
    from unittest.mock import patch

    import httpx
    from anthropic import APIStatusError

    from claude_openai_proxy.app import chat_completions
    from claude_openai_proxy.models import ChatCompletionRequest, ChatMessage

    request = ChatCompletionRequest(
        model="claude-sonnet-4-6",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    mock_raw_request = AsyncMock()
    mock_raw_request.is_disconnected = AsyncMock(return_value=False)

    mock_response = httpx.Response(
        status_code=429, request=httpx.Request("POST", "https://example.com")
    )
    exc = APIStatusError(message="rate limited", response=mock_response, body=None)

    with patch("claude_openai_proxy.app.create_message", side_effect=exc):
        response = await chat_completions(request, mock_raw_request)

    assert response.status_code == 429
