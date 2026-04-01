# Release

Release checklist:

1. Update version in `pyproject.toml` and `src/neat_optim/_version.py`
2. Update `CHANGELOG.md`
3. Run `ruff check .`, `pytest`, and `python -m build`
4. Tag the release
5. Publish to PyPI
6. Publish GitHub release notes
