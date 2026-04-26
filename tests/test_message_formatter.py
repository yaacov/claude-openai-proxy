"""Tests for OpenAI → Anthropic message conversion."""

from __future__ import annotations

import json

from claude_openai_proxy.message_formatter import (
    extract_system_message,
    messages_to_anthropic,
)
from claude_openai_proxy.models import ChatMessage


def test_extract_system_message():
    messages = [
        ChatMessage(role="system", content="Be helpful"),
        ChatMessage(role="user", content="Hi"),
    ]
    assert extract_system_message(messages) == "Be helpful"


def test_extract_system_message_none():
    messages = [ChatMessage(role="user", content="Hi")]
    assert extract_system_message(messages) is None


def test_simple_user_assistant():
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
        ChatMessage(role="user", content="How are you?"),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "assistant", "content": "Hi there!"}
    assert result[2] == {"role": "user", "content": "How are you?"}


def test_system_messages_are_skipped():
    messages = [
        ChatMessage(role="system", content="Be helpful"),
        ChatMessage(role="user", content="Hi"),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"


def test_assistant_with_tool_calls():
    messages = [
        ChatMessage(role="user", content="What's the weather?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "SF"}),
                    },
                }
            ],
        ),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    content = result[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "tool_use"
    assert content[0]["name"] == "get_weather"
    assert content[0]["input"] == {"location": "SF"}


def test_tool_results_grouped_in_user_message():
    messages = [
        ChatMessage(role="user", content="What's the weather?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "SF"}),
                    },
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", content="72°F and sunny"),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 3
    assert result[2]["role"] == "user"
    content = result[2]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "call_1"
    assert content[0]["content"] == "72°F and sunny"


def test_consecutive_same_role_merged():
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="user", content="Are you there?"),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2


def test_assistant_with_content_and_tool_calls():
    """Assistant message with both text content and tool_calls produces
    a list with the text block first, then tool_use blocks."""
    messages = [
        ChatMessage(role="user", content="What's the weather?"),
        ChatMessage(
            role="assistant",
            content="Let me check the weather for you.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "SF"}),
                    },
                }
            ],
        ),
    ]
    result = messages_to_anthropic(messages)
    assert len(result) == 2
    content = result[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Let me check the weather for you."}
    assert content[1]["type"] == "tool_use"
    assert content[1]["name"] == "get_weather"
    assert content[1]["input"] == {"location": "SF"}


def test_assistant_first_gets_user_prefix():
    messages = [
        ChatMessage(role="assistant", content="I was already talking"),
        ChatMessage(role="user", content="Continue"),
    ]
    result = messages_to_anthropic(messages)
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "user"
