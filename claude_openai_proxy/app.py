"""FastAPI application: OpenAI-compatible API backed by Claude Code CLI."""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI
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
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "sonnet",
    "haiku",
    "opus",
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


# ── Chat Completions ────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    client_system = extract_system_message(request.messages)
    system_prompt = build_system_prompt(client_system, tools=request.tools)
    prompt = messages_to_prompt(request.messages)

    if not prompt.strip():
        prompt = "(empty)"

    model = normalize_model(request.model)

    lines = spawn_cli(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
    )

    try:
        if request.stream:
            return StreamingResponse(
                build_streaming_response(lines, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await build_complete_response(lines, request.model)
        return response

    except Exception as exc:
        error = ErrorResponse(error=ErrorDetail(message=str(exc), type="server_error"))
        return JSONResponse(status_code=500, content=error.model_dump())
