"""OpenAI-compatible request/response Pydantic models."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

# ── Request models ──────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class FunctionSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    type: str = "function"
    function: FunctionSpec


class ChatCompletionRequest(BaseModel):
    model: str = "sonnet"
    messages: list[ChatMessage]
    stream: bool = False
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Accepted but ignored (Claude CLI doesn't expose these)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: bool | None = None


# ── Response models (non-streaming) ─────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ── Response models (streaming) ─────────────────────────────────────────────


class DeltaToolCall(BaseModel):
    index: int = 0
    id: str | None = None
    type: str | None = None
    function: ToolCallFunction | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]


# ── Error model ─────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    message: str
    type: str = "server_error"
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
