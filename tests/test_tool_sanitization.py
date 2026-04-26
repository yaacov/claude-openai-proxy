"""Tests for tool name normalization, validation, and deduplication."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from claude_openai_proxy.models import ToolCall, ToolCallFunction
from claude_openai_proxy.response_builder import (
    _deduplicate,
    _filter_valid,
    build_complete_response,
)
from claude_openai_proxy.tool_parser import normalize_tool_name

# ── normalize_tool_name ─────────────────────────────────────────────────────


class TestNormalizeToolName:
    def test_strips_mcp_prefix(self):
        assert normalize_tool_name("mcp__mtv__mtv_read") == "mtv_read"

    def test_strips_mcp_prefix_metrics(self):
        assert normalize_tool_name("mcp__metrics__metrics_read") == "metrics_read"

    def test_strips_mcp_prefix_debug(self):
        assert normalize_tool_name("mcp__debug__debug_read") == "debug_read"

    def test_preserves_plain_name(self):
        assert normalize_tool_name("mtv_read") == "mtv_read"

    def test_preserves_empty_string(self):
        assert normalize_tool_name("") == ""

    def test_preserves_name_with_underscores(self):
        assert normalize_tool_name("my_cool_tool") == "my_cool_tool"

    def test_only_strips_leading_mcp_prefix(self):
        assert normalize_tool_name("not_mcp__foo__bar") == "not_mcp__foo__bar"

    def test_strips_single_segment_server_name(self):
        assert normalize_tool_name("mcp__server__tool") == "tool"

    def test_strips_underscored_server_name(self):
        assert normalize_tool_name("mcp__my_server__tool") == "tool"

    def test_strips_multi_underscore_server_name(self):
        assert normalize_tool_name("mcp__my_cool_server__read") == "read"


# ── _filter_valid ───────────────────────────────────────────────────────────


def _make_call(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        function=ToolCallFunction(name=name, arguments=args),
    )


class TestFilterValid:
    def test_keeps_valid_calls(self):
        calls = [_make_call("mtv_read"), _make_call("debug_read")]
        valid = {"mtv_read", "debug_read", "metrics_read"}
        result = _filter_valid(calls, valid)
        assert len(result) == 2

    def test_drops_hallucinated_calls(self):
        calls = [_make_call("mtv_read"), _make_call("set_context")]
        valid = {"mtv_read", "debug_read"}
        result = _filter_valid(calls, valid)
        assert len(result) == 1
        assert result[0].function.name == "mtv_read"

    def test_no_filter_when_valid_names_is_none(self):
        calls = [_make_call("anything"), _make_call("goes")]
        result = _filter_valid(calls, None)
        assert len(result) == 2

    def test_empty_calls(self):
        result = _filter_valid([], {"mtv_read"})
        assert result == []


# ── _deduplicate ────────────────────────────────────────────────────────────


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        args = json.dumps({"command": "health"})
        calls = [
            _make_call("mtv_read", args),
            _make_call("mtv_read", args),
        ]
        result = _deduplicate(calls)
        assert len(result) == 1
        assert result[0].function.name == "mtv_read"

    def test_keeps_different_arguments(self):
        calls = [
            _make_call("mtv_read", json.dumps({"command": "health"})),
            _make_call("mtv_read", json.dumps({"command": "get provider"})),
        ]
        result = _deduplicate(calls)
        assert len(result) == 2

    def test_keeps_different_names(self):
        args = json.dumps({"command": "health"})
        calls = [_make_call("mtv_read", args), _make_call("debug_read", args)]
        result = _deduplicate(calls)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate([]) == []


# ── Integration: build_complete_response ────────────────────────────────────


def _make_anthropic_message(
    content_blocks: list,
    model: str = "test-model",
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    """Build a mock Anthropic Message for testing."""
    message = MagicMock()
    message.id = "msg_test123"
    message.model = model

    blocks = []
    for block_def in content_blocks:
        block = SimpleNamespace(**block_def)
        blocks.append(block)
    message.content = blocks

    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    message.usage = usage
    return message


def test_complete_response_text_only():
    """Simple text response is converted correctly."""
    msg = _make_anthropic_message([{"type": "text", "text": "Hello!"}])
    resp = build_complete_response(msg, "test-model")
    assert resp.choices[0].message.content == "Hello!"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.tool_calls is None


def test_complete_response_tool_use():
    """Tool-use blocks are converted to OpenAI tool_calls."""
    msg = _make_anthropic_message(
        [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "mtv_read",
                "input": {"command": "health"},
            }
        ]
    )
    resp = build_complete_response(msg, "test-model")
    assert resp.choices[0].finish_reason == "tool_calls"
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "mtv_read"
    assert json.loads(tool_calls[0].function.arguments) == {"command": "health"}


def test_complete_response_normalizes_and_validates():
    """mcp__ prefix is stripped and hallucinated tool is dropped."""
    msg = _make_anthropic_message(
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "mcp__mtv__mtv_read",
                "input": {"command": "health"},
            },
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "set_context",
                "input": {"key": "ns"},
            },
        ]
    )
    resp = build_complete_response(msg, "test-model", valid_tool_names={"mtv_read"})
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "mtv_read"
    assert resp.choices[0].finish_reason == "tool_calls"


def test_complete_response_deduplicates():
    """Duplicate tool calls with same name+args are collapsed."""
    msg = _make_anthropic_message(
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "mtv_read",
                "input": {"command": "health"},
            },
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "mtv_read",
                "input": {"command": "health"},
            },
        ]
    )
    resp = build_complete_response(msg, "test-model", valid_tool_names={"mtv_read"})
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1


def test_complete_response_no_tools_passes_through():
    """When valid_tool_names is None, no filtering occurs."""
    msg = _make_anthropic_message(
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "any_tool",
                "input": {"x": 1},
            }
        ]
    )
    resp = build_complete_response(msg, "test-model", valid_tool_names=None)
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "any_tool"


def test_complete_response_usage():
    """Usage tokens are mapped correctly."""
    msg = _make_anthropic_message(
        [{"type": "text", "text": "hi"}],
        input_tokens=100,
        output_tokens=50,
    )
    resp = build_complete_response(msg, "test-model")
    assert resp.usage.prompt_tokens == 100
    assert resp.usage.completion_tokens == 50
    assert resp.usage.total_tokens == 150
