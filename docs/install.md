# Installation

All methods require the **Claude Code CLI** (`claude`) to be installed and
authenticated — the binary must be on your `PATH`.

Python **3.11+** is required.

## Option 1 — pip (from PyPI)

```bash
pip install --user claude-openai-proxy
```

Make sure `~/.local/bin` is in your `PATH`, then run:

```bash
claude-openai-proxy
```

## Option 2 — uv tool (isolated install)

[`uv tool`](https://docs.astral.sh/uv/concepts/tools/) installs the package
into its own isolated virtual environment while placing the CLI on your `PATH`:

```bash
uv tool install claude-openai-proxy
```

Run the proxy:

```bash
claude-openai-proxy
```

To upgrade later:

```bash
uv tool upgrade claude-openai-proxy
```

## Option 3 — pipx (isolated install)

[`pipx`](https://pipx.pypa.io/) works the same way as `uv tool` — an isolated
environment with the CLI exposed on your `PATH`:

```bash
pipx install claude-openai-proxy
```

## Option 4 — From source

Clone the repository and use **uv** to sync dependencies:

```bash
git clone https://github.com/yaacov/claude-openai-proxy.git
cd claude-openai-proxy
uv sync
```

Run the proxy from source:

```bash
uv run python main.py
```

## Verifying the installation

Regardless of the method, you can confirm everything is working with:

```bash
claude-openai-proxy              # starts on localhost:1234
curl http://localhost:1234/health
```
