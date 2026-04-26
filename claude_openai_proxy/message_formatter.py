"""Convert OpenAI messages to Anthropic message format.

Handles the structural differences between the two APIs:
  - System messages are extracted for the separate ``system`` parameter.
  - Assistant ``tool_calls`` become ``tool_use`` content blocks.
  - Tool-result messages become ``tool_result`` blocks inside user messages.
  - Consecutive same-role messages are merged (Anthropic requires alternation).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from claude_openai_proxy.models import ChatMessage

logger = logging.getLogger(__name__)


def extract_system_message(messages: list[ChatMessage]) -> str | None:
    """Return the content of the first system message, or ``None``."""
    for message in messages:
        if message.role == "system":
            return message.content
    return None


def _convert_tool_calls_to_blocks(tool_calls: list[dict[str, Any]]) -> list[dict]:
    """Convert OpenAI-style tool_calls to Anthropic tool_use content blocks."""
    blocks: list[dict] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_raw

        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": args,
            }
        )
    return blocks


def _merge_consecutive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role messages (Anthropic requires alternation)."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]

            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(curr_content, str):
                curr_content = [{"type": "text", "text": curr_content}]

            merged[-1]["content"] = prev_content + curr_content
        else:
            merged.append(msg)
    return merged


def messages_to_anthropic(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages to Anthropic message format.

    System messages are skipped (handled separately via ``extract_system_message``).
    The result list has alternating user/assistant roles as Anthropic requires.
    """
    result: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            continue

        if message.role == "user":
            result.append({"role": "user", "content": message.content or ""})

        elif message.role == "assistant":
            if message.tool_calls:
                content: list[dict] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                content.extend(_convert_tool_calls_to_blocks(message.tool_calls))
                result.append({"role": "assistant", "content": content})
            else:
                result.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                    }
                )

        elif message.role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content or "",
            }
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
            ):
                result[-1]["content"].append(tool_result)
            else:
                result.append({"role": "user", "content": [tool_result]})

    result = _merge_consecutive(result)

    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": "(continued)"})

    return result
