"""Translate Anthropic SDK responses into OpenAI-compatible response objects.

Two modes:
  - **Non-streaming**: convert an Anthropic ``Message`` to a single
    ``ChatCompletionResponse``.
  - **Streaming**: convert raw Anthropic stream events into
    ``data: <json>\\n\\n`` SSE strings.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from claude_openai_proxy.models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    DeltaMessage,
    DeltaToolCall,
    ResponseMessage,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types import Message

logger = logging.getLogger(__name__)

_MCP_PREFIX_RE = re.compile(r"^mcp__.+?__")


def _normalize_tool_name(name: str) -> str:
    """Strip ``mcp__<server>__`` prefixes Claude sometimes hallucinates."""
    cleaned = _MCP_PREFIX_RE.sub("", name)
    if cleaned != name:
        logger.info("Normalized tool name: %r -> %r", name, cleaned)
    return cleaned


def _filter_valid(
    calls: list[ToolCall],
    valid_names: set[str] | None,
) -> list[ToolCall]:
    """Drop tool calls whose name doesn't match any tool the client sent."""
    if not valid_names or not calls:
        return calls
    kept: list[ToolCall] = []
    for call in calls:
        if call.function.name in valid_names:
            kept.append(call)
        else:
            logger.warning("Dropping hallucinated tool call: %r", call.function.name)
    return kept


def _deduplicate(calls: list[ToolCall]) -> list[ToolCall]:
    """Remove duplicate tool calls sharing the same (name, arguments)."""
    seen: set[tuple[str, str]] = set()
    unique: list[ToolCall] = []
    for call in calls:
        key = (call.function.name, call.function.arguments)
        if key not in seen:
            seen.add(key)
            unique.append(call)
        else:
            logger.info(
                "Deduplicated tool call: %s(%s)",
                call.function.name,
                call.function.arguments[:80],
            )
    return unique


# ── Non-streaming ───────────────────────────────────────────────────────────


def build_complete_response(
    message: Message,
    request_model: str,
    valid_tool_names: set[str] | None = None,
) -> ChatCompletionResponse:
    """Convert an Anthropic ``Message`` into a ``ChatCompletionResponse``."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    function=ToolCallFunction(
                        name=_normalize_tool_name(block.name),
                        arguments=json.dumps(block.input),
                    ),
                )
            )

    tool_calls = _filter_valid(tool_calls, valid_tool_names)
    tool_calls = _deduplicate(tool_calls)

    content = "".join(text_parts) or None
    has_tools = len(tool_calls) > 0
    finish_reason = "tool_calls" if has_tools else "stop"

    usage = UsageInfo(
        prompt_tokens=message.usage.input_tokens,
        completion_tokens=message.usage.output_tokens,
        total_tokens=message.usage.input_tokens + message.usage.output_tokens,
    )

    logger.info(
        "Complete response: %d text blocks, %d tool calls",
        len(text_parts),
        len(tool_calls),
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{message.id}",
        created=int(time.time()),
        model=message.model or request_model,
        choices=[
            Choice(
                message=ResponseMessage(
                    content=content,
                    tool_calls=tool_calls if has_tools else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


# ── Streaming ───────────────────────────────────────────────────────────────


async def build_streaming_response(
    events: AsyncIterator[Any],
    request_model: str,
    valid_tool_names: set[str] | None = None,
) -> AsyncIterator[str]:
    """Yield ``data: <json>\\n\\n`` SSE strings from raw Anthropic stream events.

    Event types handled:
      - ``message_start``         → emit role chunk
      - ``content_block_start``   → emit tool_call header (for tool_use blocks)
      - ``content_block_delta``   → emit text content or tool arguments
      - ``message_delta``         → emit finish_reason chunk + ``[DONE]``
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request_model
    sent_role = False
    sent_done = False

    # Map content-block index → OpenAI tool_call index (O(1) lookup)
    tool_block_indices: dict[int, int] = {}
    tool_call_counter = 0

    async for event in events:
        etype = event.type

        if etype == "message_start":
            if hasattr(event, "message") and event.message:
                model = event.message.model or request_model
                chunk_id = f"chatcmpl-{event.message.id}"

            if not sent_role:
                yield _sse(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
                    )
                )
                sent_role = True

        elif etype == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                name = _normalize_tool_name(block.name)
                if valid_tool_names and name not in valid_tool_names:
                    continue
                idx = tool_call_counter
                tool_block_indices[event.index] = idx
                tool_call_counter += 1
                yield _sse(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[
                            StreamChoice(
                                delta=DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=idx,
                                            id=block.id,
                                            type="function",
                                            function=ToolCallFunction(
                                                name=name, arguments=""
                                            ),
                                        )
                                    ]
                                )
                            )
                        ],
                    )
                )

        elif etype == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                yield _sse(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[StreamChoice(delta=DeltaMessage(content=delta.text))],
                    )
                )
            elif delta.type == "input_json_delta":
                if event.index not in tool_block_indices:
                    continue
                idx = tool_block_indices[event.index]
                yield _sse(
                    ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[
                            StreamChoice(
                                delta=DeltaMessage(
                                    tool_calls=[
                                        DeltaToolCall(
                                            index=idx,
                                            function=ToolCallFunction(
                                                name="",
                                                arguments=delta.partial_json,
                                            ),
                                        )
                                    ]
                                )
                            )
                        ],
                    )
                )

        elif etype == "message_delta":
            stop_reason = getattr(event.delta, "stop_reason", None)
            finish = "tool_calls" if stop_reason == "tool_use" else "stop"
            yield _sse(
                ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish)],
                )
            )
            yield "data: [DONE]\n\n"
            sent_done = True

    if not sent_done:
        yield "data: [DONE]\n\n"


def _sse(chunk: ChatCompletionChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"
