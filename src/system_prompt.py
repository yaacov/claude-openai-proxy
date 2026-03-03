"""Build the --system-prompt value with the LLM behavioral shim."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import ToolSpec

_PREAMBLE = """\
You are a language model serving API requests. You are NOT an interactive \
coding agent. You do not have access to a filesystem, terminal, or any \
built-in tools. You operate purely as a text-in/text-out LLM.

When the user's instructions define tools, treat them as YOUR tools that \
you can invoke. You DO have these tools. Use them when appropriate.

Do not wrap tool calls in markdown code fences. \
Do not explain that you are going to make a tool call -- just make it."""

_TOOL_CALL_FORMAT = """\
CRITICAL: When you want to call a tool, you MUST use this EXACT format \
(one call per block, you may make multiple calls):

<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}
</tool_call>

You MUST use <tool_call> tags. Do NOT output tool calls in any other format. \
Do NOT use markdown code fences for tool calls. \
Each <tool_call> block must contain valid JSON with "name" and "arguments" keys."""

_POSTAMBLE = """\
Remember: follow the above instructions precisely. When using tools, \
output ONLY <tool_call> blocks. Do not add preamble or disclaimers \
around tool invocations."""

_DEFAULT_CLIENT_PROMPT = "You are a helpful assistant."


def _format_tool_definitions(tools: list[ToolSpec]) -> str:
    """Render OpenAI tool specs into a block for the prompt."""
    lines = ["You have access to the following tools:\n"]
    for tool in tools:
        function = tool.function
        lines.append(f"### {function.name}")
        if function.description:
            lines.append(function.description)
        if function.parameters:
            lines.append(f"Parameters: {json.dumps(function.parameters)}")
        lines.append("")
    return "\n".join(lines)


def build_system_prompt(
    client_system_prompt: str | None,
    tools: list[ToolSpec] | None = None,
) -> str:
    """Wrap the client's system prompt with the behavioral shim.

    If *client_system_prompt* is ``None`` or empty, a minimal default is used.
    If *tools* is provided, tool definitions and the structured output format
    instructions are appended.
    """
    body = (client_system_prompt or "").strip() or _DEFAULT_CLIENT_PROMPT

    parts = [_PREAMBLE, "---", body]

    if tools:
        parts.extend(["---", _format_tool_definitions(tools), _TOOL_CALL_FORMAT])

    parts.extend(["---", _POSTAMBLE])

    return "\n\n".join(parts)
