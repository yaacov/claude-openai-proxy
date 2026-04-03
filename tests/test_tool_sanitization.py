"""Tests for tool name normalization, validation, and deduplication."""

from __future__ import annotations

import json

import pytest

from claude_openai_proxy.models import ToolCall, ToolCallFunction
from claude_openai_proxy.response_builder import (
    _deduplicate_tool_calls,
    _filter_valid_tool_calls,
    build_complete_response,
)
from claude_openai_proxy.tool_parser import extract_tool_calls, normalize_tool_name

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


# ── extract_tool_calls with normalization ───────────────────────────────────


class TestExtractToolCallsNormalization:
    def test_normalizes_mcp_prefix_in_xml_tags(self):
        text = (
            "<tool_call>\n"
            '{"name": "mcp__mtv__mtv_read", '
            '"arguments": {"command": "health"}}\n'
            "</tool_call>"
        )
        remaining, calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function.name == "mtv_read"
        assert remaining == ""

    def test_preserves_correct_name(self):
        text = (
            "<tool_call>\n"
            '{"name": "mtv_read", "arguments": {"command": "health"}}\n'
            "</tool_call>"
        )
        _, calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].function.name == "mtv_read"


# ── _filter_valid_tool_calls ────────────────────────────────────────────────


def _make_call(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        function=ToolCallFunction(name=name, arguments=args),
    )


class TestFilterValidToolCalls:
    def test_keeps_valid_calls(self):
        calls = [_make_call("mtv_read"), _make_call("debug_read")]
        valid = {"mtv_read", "debug_read", "metrics_read"}
        result = _filter_valid_tool_calls(calls, valid)
        assert len(result) == 2

    def test_drops_hallucinated_calls(self):
        calls = [_make_call("mtv_read"), _make_call("set_context")]
        valid = {"mtv_read", "debug_read"}
        result = _filter_valid_tool_calls(calls, valid)
        assert len(result) == 1
        assert result[0].function.name == "mtv_read"

    def test_no_filter_when_valid_names_is_none(self):
        calls = [_make_call("anything"), _make_call("goes")]
        result = _filter_valid_tool_calls(calls, None)
        assert len(result) == 2

    def test_empty_calls(self):
        result = _filter_valid_tool_calls([], {"mtv_read"})
        assert result == []


# ── _deduplicate_tool_calls ─────────────────────────────────────────────────


class TestDeduplicateToolCalls:
    def test_removes_exact_duplicates(self):
        args = json.dumps({"command": "health"})
        calls = [
            _make_call("mtv_read", args),
            _make_call("mtv_read", args),
        ]
        result = _deduplicate_tool_calls(calls)
        assert len(result) == 1
        assert result[0].function.name == "mtv_read"

    def test_keeps_different_arguments(self):
        calls = [
            _make_call("mtv_read", json.dumps({"command": "health"})),
            _make_call("mtv_read", json.dumps({"command": "get provider"})),
        ]
        result = _deduplicate_tool_calls(calls)
        assert len(result) == 2

    def test_keeps_different_names(self):
        args = json.dumps({"command": "health"})
        calls = [_make_call("mtv_read", args), _make_call("debug_read", args)]
        result = _deduplicate_tool_calls(calls)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate_tool_calls([]) == []


# ── Integration: build_complete_response ────────────────────────────────────


async def _lines_from(items: list[str]):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_complete_response_normalizes_and_validates():
    """End-to-end: mcp__ prefix is stripped and hallucinated tool is dropped."""
    events = [
        json.dumps({"type": "system", "session_id": "s1", "model": "test"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<tool_call>\n"
                                '{"name": "mcp__mtv__mtv_read", '
                                '"arguments": {"command": "health"}}\n'
                                "</tool_call>\n"
                                "<tool_call>\n"
                                '{"name": "set_context", '
                                '"arguments": {"key": "ns"}}\n'
                                "</tool_call>"
                            ),
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "result": "",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        ),
    ]

    resp = await build_complete_response(
        _lines_from(events), "test-model", valid_tool_names={"mtv_read"}
    )
    assert len(resp.choices) == 1
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "mtv_read"
    assert resp.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_complete_response_deduplicates():
    """Duplicate tool calls with same name+args are collapsed."""
    tc = (
        "<tool_call>\n"
        '{"name": "mtv_read", "arguments": {"command": "health"}}\n'
        "</tool_call>"
    )
    events = [
        json.dumps({"type": "system", "session_id": "s1", "model": "test"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": f"{tc}\n{tc}"}]},
            }
        ),
        json.dumps(
            {
                "type": "result",
                "result": "",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }
        ),
    ]

    resp = await build_complete_response(
        _lines_from(events), "test-model", valid_tool_names={"mtv_read"}
    )
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1


@pytest.mark.asyncio
async def test_complete_response_no_tools_passes_through():
    """When valid_tool_names is None, no filtering occurs."""
    events = [
        json.dumps({"type": "system", "session_id": "s1", "model": "test"}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<tool_call>\n"
                                '{"name": "any_tool", '
                                '"arguments": {"x": 1}}\n'
                                "</tool_call>"
                            ),
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "result": "",
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }
        ),
    ]

    resp = await build_complete_response(
        _lines_from(events), "test-model", valid_tool_names=None
    )
    tool_calls = resp.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "any_tool"
