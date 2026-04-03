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
pip install --user claude-openai-proxy
```

Make sure `~/.local/bin` is in your `PATH`.

> More options (uv tool, pipx, from source) are covered in the
> [installation guide](docs/install.md).

## Run

```bash
claude-openai-proxy                          # localhost:1234
HOST=0.0.0.0 PORT=5000 claude-openai-proxy   # all interfaces, custom port
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

### Test

```bash
uv run pytest               # run all tests
uv run pytest -v            # verbose output
```

## Timeouts & Disconnect Handling

The proxy guards against runaway requests with two independent timeouts and
automatically cleans up when a client disconnects.

| Guard | What it does |
|---|---|
| **Per-line timeout** (`CLAUDE_TIMEOUT`) | Kills the CLI if a single line of output takes longer than this many seconds. Catches hung processes. |
| **Total request timeout** (`CLAUDE_MAX_REQUEST_TIMEOUT`) | Kills the CLI once the overall wall-clock time for the request exceeds this limit, regardless of per-line activity. |
| **Client disconnect** | For both streaming and non-streaming responses the proxy polls for client disconnection and stops the underlying work immediately. |

## Environment Variables

| Variable                    | Default     | Description                                       |
|-----------------------------|-------------|---------------------------------------------------|
| `HOST`                      | `127.0.0.1` | Server listen address                             |
| `PORT`                      | `1234`      | Server listen port                                |
| `CLAUDE_TIMEOUT`            | `300`       | Max seconds to wait for a single line of output   |
| `CLAUDE_MAX_REQUEST_TIMEOUT`| `600`       | Max total wall-clock seconds per request          |
| `DISABLE_BUILTIN_TOOLS`     | `1`         | Set to `0` to keep Claude CLI built-in tools enabled |
