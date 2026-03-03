"""Convert an OpenAI messages array into a Claude CLI prompt string.

Responsibilities:
  1. Extract the system message (if any) for separate --system-prompt handling.
  2. Serialize the remaining user / assistant messages into a single prompt
     string that Claude receives via ``-p``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import ChatMessage

_ROLE_LABELS = {
    "user": "[User]",
    "assistant": "[Assistant]",
    "tool": "[Tool Result]",
}


def extract_system_message(messages: list[ChatMessage]) -> str | None:
    """Return the content of the first system message, or ``None``."""
    for message in messages:
        if message.role == "system":
            return message.content
    return None


def _format_assistant_message(message: ChatMessage) -> str:
    """Format an assistant message, including any tool_calls."""
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function = tool_call.get("function", {})
            parts.append(
                json.dumps(
                    {
                        "tool_call": {
                            "id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    }
                )
            )
    return " ".join(parts) if parts else ""


def _format_tool_message(message: ChatMessage) -> str:
    """Format a tool-result message."""
    result = {"tool_call_id": message.tool_call_id, "result": message.content or ""}
    return json.dumps(result)


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Serialize non-system messages into a labelled prompt string.

    The last user message is always included.  Earlier turns provide
    conversational context so Claude can continue coherently.
    """
    parts: list[str] = []
    for message in messages:
        if message.role == "system":
            continue

        label = _ROLE_LABELS.get(message.role, f"[{message.role.title()}]")

        if message.role == "assistant" and message.tool_calls:
            content = _format_assistant_message(message)
        elif message.role == "tool":
            content = _format_tool_message(message)
        else:
            content = message.content or ""

        parts.append(f"{label}: {content}")

    return "\n\n".join(parts)
