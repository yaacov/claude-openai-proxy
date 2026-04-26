"""Build the system prompt passed to the Anthropic API."""

from __future__ import annotations

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_system_prompt(client_system_prompt: str | None) -> str | None:
    """Return the caller's system prompt, falling back to a sensible default.

    - ``None`` → ``_DEFAULT_SYSTEM_PROMPT`` (caller didn't send one).
    - Empty / whitespace-only string → ``None`` (explicit opt-out,
      lets the Anthropic API use its own default behaviour).
    - Non-empty string → returned as-is (stripped).
    """
    if client_system_prompt is None:
        return _DEFAULT_SYSTEM_PROMPT
    stripped = client_system_prompt.strip()
    return stripped if stripped else None
