#!/usr/bin/env bash
set -euo pipefail

ruff check .
pytest
python -m build
