"""FastAPI application: OpenAI-compatible API backed by Anthropic Vertex AI."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from anthropic import APIStatusError, APITimeoutError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from claude_openai_proxy.anthropic_client import create_message, stream_message
from claude_openai_proxy.message_formatter import (
    extract_system_message,
    messages_to_anthropic,
)
from claude_openai_proxy.models import (
    ChatCompletionRequest,
    ErrorDetail,
    ErrorResponse,
)
from claude_openai_proxy.response_builder import (
    build_complete_response,
    build_streaming_response,
)
from claude_openai_proxy.system_prompt import build_system_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_openai_proxy.models import ToolSpec

logger = logging.getLogger(__name__)

app = FastAPI(title="Claude OpenAI Proxy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]


def normalize_model(raw: str) -> str:
    """Strip date-pin suffixes (e.g. ``@20250805``) that some clients add."""
    name = re.sub(r"@\d+$", "", raw).strip()
    if name != raw:
        logger.info("Stripped date suffix: %r -> %r", raw, name)
    return name


def _convert_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
    """Convert OpenAI tool specs to Anthropic tool format."""
    if not tools:
        return None
    return [
        {
            "name": t.function.name,
            "description": t.function.description or "",
            "input_schema": t.function.parameters
            or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def _sanitize_error(message: str) -> str:
    """Strip internal details from upstream error messages."""
    return message[:500] if message else "Unknown error"


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Models ──────────────────────────────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            }
            for model in AVAILABLE_MODELS
        ],
    }


@app.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str):
    name = normalize_model(model_id)
    if name not in AVAILABLE_MODELS:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=f"The model '{model_id}' does not exist",
                    type="invalid_request_error",
                    code="model_not_found",
                )
            ).model_dump(),
        )
    return {
        "id": name,
        "object": "model",
        "created": 1700000000,
        "owned_by": "anthropic",
    }


# ── Disconnect helper ──────────────────────────────────────────────────────


async def _streaming_with_disconnect(
    inner: AsyncIterator[str], raw_request: Request
) -> AsyncIterator[str]:
    """Wrap a streaming generator so we stop when the client disconnects."""
    async for chunk in inner:
        if await raw_request.is_disconnected():
            logger.warning("Client disconnected during streaming response")
            break
        yield chunk


# ── Chat Completions ────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    client_system = extract_system_message(request.messages)
    system_prompt = build_system_prompt(client_system)
    anthropic_messages = messages_to_anthropic(request.messages)

    model = normalize_model(request.model)
    max_tokens = request.max_tokens or 8192

    anthropic_tools = _convert_tools(request.tools)

    valid_tool_names: set[str] | None = None
    if anthropic_tools:
        valid_tool_names = {t["name"] for t in anthropic_tools}

    try:
        if request.stream:
            events = stream_message(
                model=model,
                messages=anthropic_messages,
                system=system_prompt,
                tools=anthropic_tools,
                max_tokens=max_tokens,
            )
            return StreamingResponse(
                _streaming_with_disconnect(
                    build_streaming_response(events, request.model, valid_tool_names),
                    raw_request,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        message = await create_message(
            model=model,
            messages=anthropic_messages,
            system=system_prompt,
            tools=anthropic_tools,
            max_tokens=max_tokens,
        )
        return build_complete_response(message, request.model, valid_tool_names)

    except asyncio.CancelledError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
    except APITimeoutError as exc:
        logger.warning("Anthropic API timeout: %s", exc)
        error = ErrorResponse(
            error=ErrorDetail(message=_sanitize_error(str(exc)), type="timeout_error")
        )
        return JSONResponse(status_code=504, content=error.model_dump())
    except APIStatusError as exc:
        logger.warning("Anthropic API error %d: %s", exc.status_code, exc.message)
        error = ErrorResponse(
            error=ErrorDetail(message=_sanitize_error(exc.message), type="api_error")
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump())
    except Exception as exc:
        logger.exception("Error in chat_completions")
        error = ErrorResponse(
            error=ErrorDetail(message=_sanitize_error(str(exc)), type="server_error")
        )
        return JSONResponse(status_code=500, content=error.model_dump())
