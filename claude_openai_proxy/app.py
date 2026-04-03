"""FastAPI application: OpenAI-compatible API backed by Claude Code CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from claude_openai_proxy.claude_cli import spawn_cli
from claude_openai_proxy.message_formatter import (
    extract_system_message,
    messages_to_prompt,
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
    """Strip date-pin suffixes (e.g. ``@20250805``) the CLI doesn't accept."""
    name = re.sub(r"@\d+$", "", raw).strip()
    if name != raw:
        logger.info("Stripped date suffix: %r -> %r", raw, name)
    return name


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


# ── Disconnect helpers ──────────────────────────────────────────────────────


async def _streaming_with_disconnect(
    inner: AsyncIterator[str], raw_request: Request
) -> AsyncIterator[str]:
    """Wrap a streaming generator so we stop when the client disconnects."""
    async for chunk in inner:
        if await raw_request.is_disconnected():
            logger.warning("Client disconnected during streaming response")
            break
        yield chunk


async def _complete_with_disconnect(
    lines: AsyncIterator[str],
    model: str,
    raw_request: Request,
    valid_tool_names: set[str] | None = None,
):
    """Run ``build_complete_response`` while polling for client disconnect."""
    task = asyncio.create_task(build_complete_response(lines, model, valid_tool_names))
    try:
        while not task.done():
            if await raw_request.is_disconnected():
                logger.warning("Client disconnected during non-streaming response")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                await lines.aclose()  # type: ignore[union-attr]
                raise asyncio.CancelledError
            await asyncio.sleep(1)
        return task.result()
    except asyncio.CancelledError:
        raise
    except BaseException:
        task.cancel()
        raise


# ── Chat Completions ────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    client_system = extract_system_message(request.messages)
    system_prompt = build_system_prompt(client_system, tools=request.tools)
    prompt = messages_to_prompt(request.messages)

    if not prompt.strip():
        prompt = "(empty)"

    model = normalize_model(request.model)

    valid_tool_names: set[str] | None = None
    if request.tools:
        valid_tool_names = {t.function.name for t in request.tools}

    lines = spawn_cli(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
    )

    try:
        if request.stream:
            return StreamingResponse(
                _streaming_with_disconnect(
                    build_streaming_response(lines, request.model, valid_tool_names),
                    raw_request,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await _complete_with_disconnect(
            lines, request.model, raw_request, valid_tool_names
        )
        return response

    except asyncio.CancelledError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
    except Exception as exc:
        error = ErrorResponse(error=ErrorDetail(message=str(exc), type="server_error"))
        return JSONResponse(status_code=500, content=error.model_dump())
