"""Tool-name normalization utilities.

With the Anthropic SDK handling native tool calling, the XML ``<tool_call>``
parsing is no longer needed.  This module retains the ``normalize_tool_name``
helper which strips ``mcp__<server>__`` prefixes that Claude sometimes
hallucinates.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MCP_PREFIX_RE = re.compile(r"^mcp__.+?__")


def normalize_tool_name(name: str) -> str:
    """Strip ``mcp__<server>__`` prefixes that Claude sometimes hallucinates.

    For example, ``mcp__mtv__mtv_read`` becomes ``mtv_read``.
    """
    cleaned = _MCP_PREFIX_RE.sub("", name)
    if cleaned != name:
        logger.info("Normalized tool name: %r -> %r", name, cleaned)
    return cleaned
