# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,keras]"
```

## Checks

```bash
ruff check .
ruff format .
pytest
python -m build
```
