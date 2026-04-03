"""Parse ``<tool_call>`` blocks from Claude's text output.

When the Claude CLI runs with ``--tools ""`` (no built-in tools), it cannot
produce structured ``tool_use`` content blocks.  Instead, the system prompt
instructs Claude to emit tool calls inside ``<tool_call>`` XML tags::

    <tool_call>
    {"name": "metrics_read", "arguments": {"command": "query"}}
    </tool_call>

This module extracts those blocks and converts them to OpenAI ``ToolCall``
objects.  Any text outside the tags is returned as regular content.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from claude_openai_proxy.models import ToolCall, ToolCallFunction

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

_MCP_PREFIX_RE = re.compile(r"^mcp__[^_]+__")


def normalize_tool_name(name: str) -> str:
    """Strip ``mcp__<server>__`` prefixes that Claude sometimes hallucinates.

    For example, ``mcp__mtv__mtv_read`` becomes ``mtv_read``.
    """
    cleaned = _MCP_PREFIX_RE.sub("", name)
    if cleaned != name:
        logger.info("Normalized tool name: %r -> %r", name, cleaned)
    return cleaned


def extract_tool_calls(
    text: str,
) -> tuple[str, list[ToolCall]]:
    """Extract ``<tool_call>`` blocks from *text*.

    Returns a tuple of (remaining_text, tool_calls).
    *remaining_text* has the ``<tool_call>`` blocks removed and is stripped of
    leading/trailing whitespace.
    """
    calls: list[ToolCall] = []
    cleaned = text

    for match in _TOOL_CALL_RE.finditer(text):
        raw_json = match.group(1)
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        name = normalize_tool_name(obj.get("name", ""))
        arguments = obj.get("arguments", {})

        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                function=ToolCallFunction(
                    name=name,
                    arguments=(
                        json.dumps(arguments)
                        if isinstance(arguments, dict)
                        else str(arguments)
                    ),
                ),
            )
        )

    if calls:
        cleaned = _TOOL_CALL_RE.sub("", text).strip()

    return cleaned, calls
