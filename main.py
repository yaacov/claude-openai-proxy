"""Entry point for the Claude OpenAI Proxy server."""

from __future__ import annotations

import logging
import os

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)-5s %(name)-20s %(message)s",
)

from src.app import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
