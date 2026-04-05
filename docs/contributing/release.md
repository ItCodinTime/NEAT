# Release

Release checklist:

1. Update version in `pyproject.toml` and `src/neat_optim/_version.py`
2. Update `CHANGELOG.md`
3. Verify project URLs, contact points, and policy files are current
4. Run `ruff check .`, `pytest`, `mkdocs build --strict`, and `python -m build`
5. Smoke-install the wheel into a fresh virtual environment
6. Tag the release
7. Publish to PyPI
8. Publish GitHub release notes
