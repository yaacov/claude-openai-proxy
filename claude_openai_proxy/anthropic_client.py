"""Anthropic Vertex AI client for Claude API calls."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropicVertex

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types import Message

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
DEFAULT_TIMEOUT = float(os.environ.get("ANTHROPIC_TIMEOUT", "600"))

_client: AsyncAnthropicVertex | None = None


def get_client() -> AsyncAnthropicVertex:
    """Return a singleton AsyncAnthropicVertex client.

    Picks up ``CLOUD_ML_REGION`` and ``ANTHROPIC_VERTEX_PROJECT_ID``
    from the environment automatically.
    """
    global _client
    if _client is None:
        region = os.environ.get("CLOUD_ML_REGION", "<unset>")
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "<unset>")
        logger.info(
            "Initializing AsyncAnthropicVertex: region=%s project=%s timeout=%s",
            region,
            project,
            DEFAULT_TIMEOUT,
        )
        try:
            client = AsyncAnthropicVertex(timeout=DEFAULT_TIMEOUT)
        except Exception:
            logger.exception("Failed to initialize AsyncAnthropicVertex")
            raise
        _client = client
    return _client


def _build_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    system: str | None,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    return kwargs


async def create_message(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Message:
    """Make a non-streaming API call and return the response Message."""
    client = get_client()
    kwargs = _build_kwargs(model, messages, system, tools, max_tokens)
    logger.info("Creating message: model=%s messages=%d", model, len(messages))
    return await client.messages.create(**kwargs)


async def stream_message(
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AsyncIterator[Any]:
    """Make a streaming API call and yield raw stream events."""
    client = get_client()
    kwargs = _build_kwargs(model, messages, system, tools, max_tokens)
    logger.info("Streaming message: model=%s messages=%d", model, len(messages))
    async with client.messages.stream(**kwargs) as stream:
        async for event in stream:
            yield event
