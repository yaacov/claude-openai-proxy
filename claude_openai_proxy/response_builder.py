"""Translate Claude Code ``stream-json`` JSONL into OpenAI response objects.

Claude emits one JSON object per line with a ``type`` field.  We care about:

- ``system``   -- init event; carries ``session_id`` and ``model``.
- ``assistant`` -- message event; ``content`` array contains ``text`` blocks
                  and (when built-in tools are enabled) ``tool_use`` blocks.
- ``result``   -- final event; carries ``usage`` and ``result`` text.

When built-in tools are disabled (``--tools ""``), Claude outputs tool calls
as ``<tool_call>`` tags in the text.  These are parsed by
:mod:`src.tool_parser` and converted to OpenAI ``tool_calls``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
from claude_openai_proxy.tool_parser import extract_tool_calls

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class _SessionMeta:
    session_id: str = ""
    model: str = ""
    created: int = field(default_factory=lambda: int(time.time()))


def _parse_line(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_text(event: dict) -> str | None:
    """Pull text content out of an assistant-message event."""
    message = event.get("message", {})
    parts: list[str] = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts) if parts else None


def _extract_native_tool_calls(event: dict) -> list[ToolCall]:
    """Pull tool_use blocks out of an assistant-message event and convert to
    OpenAI ``tool_calls`` format."""
    message = event.get("message", {})
    calls: list[ToolCall] = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    function=ToolCallFunction(
                        name=block.get("name", ""),
                        arguments=json.dumps(block.get("input", {})),
                    ),
                )
            )
    return calls


def _extract_usage(event: dict) -> UsageInfo:
    usage = event.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


# ── Non-streaming: collect everything, return one response ──────────────────


async def build_complete_response(
    lines: AsyncIterator[str],
    request_model: str,
) -> ChatCompletionResponse:
    """Consume all JSONL lines and return a single ``ChatCompletionResponse``."""
    meta = _SessionMeta()
    text_parts: list[str] = []
    all_tool_calls: list[ToolCall] = []
    usage = UsageInfo()

    async for raw in lines:
        event = _parse_line(raw)
        if event is None:
            continue

        event_type = event.get("type")
        logger.debug("collect event type=%s", event_type)

        if event_type == "system":
            meta.session_id = event.get("session_id", meta.session_id)
            meta.model = event.get("model", meta.model)

        elif event_type == "assistant":
            chunk = _extract_text(event)
            if chunk:
                text_parts.append(chunk)

            calls = _extract_native_tool_calls(event)
            if calls:
                all_tool_calls.extend(calls)
                logger.debug("Extracted %d structured tool call(s)", len(calls))

        elif event_type == "result":
            usage = _extract_usage(event)
            result_text = event.get("result", "")
            if result_text and not text_parts:
                text_parts.append(result_text)

    full_text = "".join(text_parts)

    # When built-in tools are disabled, Claude embeds tool calls as
    # <tool_call> tags in the text.  Parse them out.
    if not all_tool_calls and full_text:
        remaining, parsed_calls = extract_tool_calls(full_text)
        if parsed_calls:
            all_tool_calls = parsed_calls
            full_text = remaining
            logger.info("Parsed %d tool call(s) from text", len(parsed_calls))

    content = full_text or None
    has_tool_calls = len(all_tool_calls) > 0
    finish_reason = "tool_calls" if has_tool_calls else "stop"

    logger.info(
        "Collected response: %d text parts, %d tool calls",
        len(text_parts),
        len(all_tool_calls),
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{meta.session_id or 'unknown'}",
        created=meta.created,
        model=meta.model or request_model,
        choices=[
            Choice(
                message=ResponseMessage(
                    content=content if content else None,
                    tool_calls=all_tool_calls if has_tool_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


# ── Streaming: yield SSE-formatted chunks ───────────────────────────────────


async def build_streaming_response(
    lines: AsyncIterator[str],
    request_model: str,
) -> AsyncIterator[str]:
    """Yield ``data: <json>\\n\\n`` SSE strings as Claude produces output.

    For streaming, structured ``tool_use`` blocks are forwarded immediately.
    Text-embedded ``<tool_call>`` tags are buffered until the ``result`` event
    so they can be parsed and emitted as proper tool_calls before ``[DONE]``.
    """
    meta = _SessionMeta()
    sent_role = False
    buffered_text: list[str] = []

    async for raw in lines:
        event = _parse_line(raw)
        if event is None:
            continue

        event_type = event.get("type")

        if event_type == "system":
            meta.session_id = event.get("session_id", meta.session_id)
            meta.model = event.get("model", meta.model)
            continue

        chunk_id = f"chatcmpl-{meta.session_id or 'unknown'}"
        model = meta.model or request_model

        if event_type == "assistant":
            if not sent_role:
                role_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=meta.created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
                )
                yield f"data: {role_chunk.model_dump_json()}\n\n"
                sent_role = True

            text = _extract_text(event)
            if text:
                buffered_text.append(text)

            calls = _extract_native_tool_calls(event)
            for index, tool_call in enumerate(calls):
                tool_call_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=meta.created,
                    model=model,
                    choices=[
                        StreamChoice(
                            delta=DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=index,
                                        id=tool_call.id,
                                        type="function",
                                        function=tool_call.function,
                                    )
                                ]
                            )
                        )
                    ],
                )
                yield f"data: {tool_call_chunk.model_dump_json()}\n\n"

        elif event_type == "result":
            if not sent_role:
                role_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=meta.created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
                )
                yield f"data: {role_chunk.model_dump_json()}\n\n"

            result_text = event.get("result", "")
            if result_text and not buffered_text:
                buffered_text.append(result_text)

            full_text = "".join(buffered_text)
            remaining, parsed_calls = extract_tool_calls(full_text)

            if remaining:
                content_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=meta.created,
                    model=model,
                    choices=[StreamChoice(delta=DeltaMessage(content=remaining))],
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"

            for index, tool_call in enumerate(parsed_calls):
                tool_call_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=meta.created,
                    model=model,
                    choices=[
                        StreamChoice(
                            delta=DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=index,
                                        id=tool_call.id,
                                        type="function",
                                        function=tool_call.function,
                                    )
                                ]
                            )
                        )
                    ],
                )
                yield f"data: {tool_call_chunk.model_dump_json()}\n\n"

            finish = "tool_calls" if parsed_calls else "stop"
            stop_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=meta.created,
                model=model,
                choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish)],
            )
            yield f"data: {stop_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
