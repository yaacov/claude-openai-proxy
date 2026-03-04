# Publishing to PyPI

Instructions for publishing `claude-openai-proxy` to PyPI under the **yaacov** account.

## Prerequisites

- A [PyPI](https://pypi.org) account
- An API token (create one at <https://pypi.org/manage/account/token/>)
- `uv` installed

## Build

```bash
uv build
```

This creates two files in `dist/`:

- `claude_openai_proxy-<version>.tar.gz` (source distribution)
- `claude_openai_proxy-<version>-py3-none-any.whl` (wheel)

## Publish

```bash
uv publish
```

You will be prompted for credentials. Use:

- **Username:** `__token__`
- **Password:** your PyPI API token (starts with `pypi-`)

To avoid the prompt, set the token as an environment variable:

```bash
export UV_PUBLISH_TOKEN=pypi-...
uv publish
```

## Version bumps

1. Update the version in `pyproject.toml` (`[project] version = "..."`)
2. Update the version in `claude_openai_proxy/app.py` (`FastAPI(..., version="...")`)
3. Rebuild and publish:

```bash
rm -rf dist/
uv build
uv publish
```

## Verify

After publishing, confirm the package is available:

```bash
pip install --user claude-openai-proxy
claude-openai-proxy --help
```

Or check the project page at <https://pypi.org/project/claude-openai-proxy/>.
