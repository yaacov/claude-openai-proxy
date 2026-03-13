"""Spawn ``claude -p`` as an async subprocess and iterate over its stdout."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "300"))
DEFAULT_MAX_REQUEST_TIMEOUT = int(os.environ.get("CLAUDE_MAX_REQUEST_TIMEOUT", "600"))
DISABLE_BUILTIN_TOOLS = os.environ.get("DISABLE_BUILTIN_TOOLS", "1") == "1"


async def spawn_cli(
    prompt: str,
    system_prompt: str,
    model: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_request_timeout: int = DEFAULT_MAX_REQUEST_TIMEOUT,
) -> AsyncIterator[str]:
    """Spawn the Claude Code CLI and yield raw stdout lines (JSONL).

    The process is always invoked with ``--output-format stream-json``
    so we get incremental JSONL.

    *timeout* caps how long we wait for a single line of output.
    *max_request_timeout* caps the total wall-clock duration of the request.

    Set ``DISABLE_BUILTIN_TOOLS=0`` to keep Claude CLI built-in tools
    (Bash, Edit, etc.) enabled.  By default they are disabled so Claude
    behaves as a pure text/tool-calling LLM.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--system-prompt",
        system_prompt,
        "--model",
        model,
    ]

    if DISABLE_BUILTIN_TOOLS:
        cmd.extend(["--tools", ""])

    logger.info("Running: %s", " ".join(cmd[:6]) + " ...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    line_count = 0
    start = time.monotonic()
    try:
        assert proc.stdout is not None
        while True:
            elapsed = time.monotonic() - start
            remaining = max_request_timeout - elapsed
            if remaining <= 0:
                logger.warning(
                    "Claude CLI total request timeout after %ds", max_request_timeout
                )
                break
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=min(timeout, remaining)
                )
            except TimeoutError:
                if time.monotonic() - start >= max_request_timeout:
                    logger.warning(
                        "Claude CLI total request timeout after %ds",
                        max_request_timeout,
                    )
                else:
                    logger.warning("Claude CLI per-line timeout after %ds", timeout)
                break
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            line_count += 1
            logger.debug("stdout[%d]: %s", line_count, decoded[:200])
            yield decoded
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()

        stderr_bytes = await proc.stderr.read() if proc.stderr else b""
        if stderr_bytes:
            logger.warning(
                "Claude CLI stderr (exit %s): %s",
                proc.returncode,
                stderr_bytes.decode("utf-8", errors="replace").strip(),
            )
        if line_count == 0:
            logger.error(
                "Claude CLI produced no stdout (exit code %s)",
                proc.returncode,
            )
