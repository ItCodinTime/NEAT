# Contributing

## Scope

NEAT is intended to be a serious optimizer library. Changes should improve one
of:

- correctness
- reproducibility
- usability
- measurable performance
- documentation quality

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,keras]"
```

If you want to exercise the Keras optimizer in `model.fit`, also install a
supported Keras backend runtime such as TensorFlow.

## Standards

- Keep the public API small.
- Do not introduce framework-specific logic into the reference engine.
- Add tests for every behavior change.
- Preserve serialization compatibility where practical.
- Benchmark before claiming a performance improvement.

## Commands

```bash
ruff check .
ruff format .
pytest
python -m build
```

## Pull Requests

- Explain the user-facing effect.
- Call out math or serialization changes explicitly.
- Include benchmark evidence for native-core or hot-path changes.

## Versioning

- `0.x`: minor versions may contain breaking changes
- `1.0+`: SemVer with deprecation windows for public APIs
