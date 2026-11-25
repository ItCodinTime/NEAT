# Release

Release checklist:

1. Update the version in `pyproject.toml` (`src/neat_optim/_version.py` reads package metadata)
2. Update `CHANGELOG.md`
3. Verify project URLs, contact points, and policy files are current
4. Run `ruff check .`, `pytest`, `mkdocs build --strict`, and `python -m build`
5. Smoke-install the wheel into a fresh virtual environment
6. Tag the release with a SemVer tag such as `v1.0.0`
7. Publish the GitHub release; CI attaches the sdist and supported wheels
8. Publish to PyPI after configuring the `pypi` trusted-publishing environment
