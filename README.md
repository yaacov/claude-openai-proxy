# Claude OpenAI Proxy

A thin Python proxy that wraps the **Anthropic Vertex AI** SDK and
exposes an **OpenAI-compatible** `/v1/chat/completions` API.

Requests are translated from the OpenAI message format to the Anthropic
format, sent to Claude via the Vertex AI API, and the responses are
converted back to OpenAI-compatible JSON (or SSE for streaming).

## Prerequisites

- Python 3.11+
- A Google Cloud project with the Anthropic Vertex AI API enabled
- Google Cloud credentials configured (e.g. `gcloud auth application-default login`)

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

## Environment Variables

| Variable                      | Default     | Description                                            |
|-------------------------------|-------------|--------------------------------------------------------|
| `HOST`                        | `127.0.0.1` | Server listen address                                  |
| `PORT`                        | `1234`      | Server listen port                                     |
| `LOG_LEVEL`                   | `INFO`      | Python logging level                                   |
| `MAX_TOKENS`                  | `8192`      | Default max tokens per response                        |
| `ANTHROPIC_TIMEOUT`           | `600`       | Request timeout in seconds for the Anthropic SDK       |
| `CLOUD_ML_REGION`             | —           | Google Cloud region (e.g. `us-east5`)                  |
| `ANTHROPIC_VERTEX_PROJECT_ID` | —           | Google Cloud project ID                                |
