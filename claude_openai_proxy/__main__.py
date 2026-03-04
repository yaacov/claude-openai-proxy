"""Allow running with ``python -m claude_openai_proxy``."""

from __future__ import annotations

import logging
import os


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)-5s %(name)-20s %(message)s",
    )

    import uvicorn

    from claude_openai_proxy.app import app

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "1234"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
