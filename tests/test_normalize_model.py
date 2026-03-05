"""Tests for the normalize_model helper."""

from claude_openai_proxy.app import normalize_model


def test_strips_date_suffix():
    assert normalize_model("claude-opus-4-6@20250805") == "claude-opus-4-6"


def test_strips_date_suffix_sonnet():
    assert normalize_model("claude-sonnet-4-5@20250101") == "claude-sonnet-4-5"


def test_preserves_full_name_without_suffix():
    assert normalize_model("claude-opus-4-6") == "claude-opus-4-6"


def test_preserves_short_alias():
    assert normalize_model("opus") == "opus"
    assert normalize_model("sonnet") == "sonnet"
    assert normalize_model("haiku") == "haiku"


def test_does_not_strip_non_date_at():
    assert normalize_model("model@abc") == "model@abc"


def test_empty_string():
    assert normalize_model("") == ""
