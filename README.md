# Claude OpenAI Proxy

A thin Python proxy that wraps the **Claude Code CLI** (`claude -p`) and
exposes an **OpenAI-compatible** `/v1/chat/completions` API.

Claude runs as a clean LLM — all built-in tools are disabled. A behavioral
shim is injected into the system prompt so Claude behaves as a stateless
text-in / text-out language model.

## Prerequisites

- **Claude Code CLI** installed and authenticated (`claude` must be on PATH)
- Python 3.11+

## Install

```bash
cd claude-openai-proxy
uv sync
```

## Run

```bash
uv run python main.py                        # localhost:8080
HOST=0.0.0.0 PORT=1234 uv run python main.py # all interfaces, custom port
```

## Endpoints

| Method | Path                    | Description                         |
|--------|-------------------------|-------------------------------------|
| GET    | `/health`               | Health check                        |
| GET    | `/v1/models`            | List available Claude models        |
| POST   | `/v1/chat/completions`  | Chat completions (streaming + sync) |

## Development

```bash
uv sync --group dev
```

### Lint

```bash
uv run ruff check .          # check for issues
uv run ruff check --fix .    # auto-fix issues
```

### Format

```bash
uv run ruff format .         # format all files
uv run ruff format --check . # check without writing
```

## Environment Variables

| Variable         | Default     | Description                          |
|------------------|-------------|--------------------------------------|
| `HOST`           | `127.0.0.1` | Server listen address                |
| `PORT`           | `8080`      | Server listen port                   |
| `CLAUDE_TIMEOUT` | `300`       | Per-request subprocess timeout (sec) |
