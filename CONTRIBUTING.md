# Contributing to NEAT

NEAT is a research-grade optimizer library. Contributions that improve correctness, reproducibility, usability, or documentation quality are welcome.

## Getting Started

### Prerequisites

- Python 3.10 or newer
- A C++ compiler (for the optional native extension — skip if you only need the NumPy path)
- A Keras backend runtime if you want to run Keras integration tests (TensorFlow or PyTorch)

### Local setup

```bash
git clone https://github.com/ItCodinTime/NEAT.git
cd NEAT
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,keras]"
pip install tensorflow            # or: pip install torch
```

---

## Development Workflow

### Running tests

```bash
pytest                           # full suite
pytest tests/unit                # unit tests only (fast, no Keras required)
pytest tests/integration         # requires Keras + a backend
pytest tests/property            # randomized invariant checks
pytest tests/regression          # reproducibility checks
pytest --cov=neat_optim --cov-report=term-missing  # with coverage
```

### Lint and format

```bash
ruff check .                     # lint (must pass in CI)
ruff format .                    # auto-format
```

### Type checking

```bash
mypy src/neat_optim              # strict mode, must pass in CI
```

### Docs build

```bash
mkdocs build --strict            # validates all internal links
mkdocs serve                     # live preview at localhost:8000
```

### Full pre-push checklist

```bash
ruff check . && ruff format --check . && mypy src/neat_optim && pytest
```

---

## Standards

**Correctness first.** The update rule in `engine/reference.py` is the canonical implementation. Any change to the algorithm must be reflected there first, and the math spec in `docs/research/math-spec.md` must be updated to match.

**Keep framework logic out of the reference engine.** `engine/reference.py` and `engine/common.py` must remain pure NumPy. Keras, TensorFlow, and PyTorch-specific code belongs in `keras_optimizer.py` or `training/`.

**Tests for every behavior change.** New hyperparameters, edge cases, and bug fixes each need a test. Native-core changes additionally need a parity test against the reference engine in `tests/unit/test_native_parity.py`.

**Serialization compatibility.** The `NEAT` Keras optimizer must remain round-trip serializable via `get_config()` / `from_config()`. Breaking this requires a minor-version bump.

**Benchmark before claiming performance improvements.** Changes to hot paths need `benchmarks/run.py` output attached to the pull request.

**Small public API surface.** Do not add public exports without a clear use case that cannot be served by existing APIs.

---

## Pull Requests

- Title: present tense, specific (`Add adaptive_preconditioning flag` not `Improvements`)
- Description: explain the **user-facing effect** and the **why**, not just the what
- Call out any changes to the math spec, `get_config`, or public API explicitly
- For native-core or hot-path changes: include `benchmarks/run.py` output
- Link to any relevant issue or math-spec discussion

---

## Reporting Bugs

Open a GitHub issue with:

1. The `neat_optim.__version__` you are running
2. Keras version (`keras.__version__`) and backend (TensorFlow / PyTorch / JAX)
3. A minimal reproducible example
4. The actual vs. expected behavior

For security issues, do not open a public issue — see [SECURITY.md](SECURITY.md).

---

## Versioning

- `0.x`: minor releases may contain breaking changes to the public API
- `1.0+`: SemVer with deprecation windows for all public APIs

Changes to `docs/research/math-spec.md` that alter the update rule increment the spec version string (`nce_spec_v1` → `nce_spec_v2`).
