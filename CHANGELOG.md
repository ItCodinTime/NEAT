# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/) from `1.0.0` onward.
Before `1.0.0`, breaking changes may occur in minor releases.

## [Unreleased]

### Added

- Optional PyTorch `TorchNEAT` adapter and modern vision, LLM, RL, diffusion,
  and large-batch benchmark runners
- Advanced cross-framework parity tests for adaptive, Lion, Nesterov,
  Lookahead, regularization, sparsity, and pruning modes
- Algorithm-phase comments and API documentation across optimizer engines and
  benchmark infrastructure
- Exact multi-seed statistics with t-distribution confidence intervals, paired
  t/Wilcoxon tests, paired Cohen's d, and strict seed-completeness validation
- Hardened benchmark manifests with source-tree, package, hardware, RNG, and
  exact-command provenance
- Default-off v3 trust-bound, opponent-ensemble, preconditioning-order,
  conflict-granularity, and deterministic reference controls with adapter tests
- Deterministic ablation and regime-map runners, mechanism-trace plotting,
  dataset checksum installation, and a named fused-native benchmark path
- Five-seed benchmark entry-point orchestration plus SGD, Lion, PCGrad, and
  CAGrad registry coverage in their applicable task settings

### Changed

- PyTorch benchmark diagnostics are sampled every ten steps to reduce
  accelerator synchronization overhead without changing updates

## [0.1.0.dev0] - 2026-04-01

### Added

**Core algorithm**
- NumPy reference implementation of the NEAT update rule (`engine/reference.py`)
- Versioned mathematical specification (`docs/research/math-spec.md`, `nce_spec_v1`)
- Conflict detection via `relu(−cos(g, o))` between current gradient and opponent proxy
- NCE correction: projection of gradient onto opponent direction, clipped to `nce_clip_ratio · ‖g‖`
- Configurable opponent sources: `momentum` (default), `previous_gradient`, `gradient_ema`, `blended`
- Correction warmup via `correction_warmup_steps`
- Conflict gating via `conflict_threshold`
- Adaptive correction scaling via `adaptive_correction` and `adaptive_correction_decay`
- Adam-style second-moment preconditioning via `adaptive_preconditioning`
- Bias correction for momentum and second moment via `bias_correction`
- Decoupled and coupled weight decay paths
- Soft L1 sparsity pressure via `sparsity_l1`
- Hard weight pruning via `prune_threshold`

**Keras integration**
- `NEAT` Keras optimizer subclass (`keras_optimizer.py`)
- Registered as Keras-serializable via `@keras.saving.register_keras_serializable`
- Full `get_config()` / `from_config()` round-trip for checkpoint compatibility
- Built-in per-step diagnostic accumulators
- `diagnostic_snapshot()` returning mean conflict ratio, correction ratio, update alignment, opponent norm, and correction active fraction
- `reset_diagnostics()` for epoch-level measurement windows

**Player-aware mode**
- `PlayerNEATConfig` for per-example gradient stepping configuration
- `create_player_states()` to initialize per-variable optimizer state from a Keras model
- `player_train_step()` to run one batch step using per-example Jacobians
- `TensorFlowPlayerStepResult` with loss, updated states, and per-player metrics

**Model compaction**
- `compact_dense_model()` — prune and retrain a dense Keras model
- `search_compact_dense_model()` — grid search over sparsity configurations
- `benchmark_inference_latency()` — measure model inference speed
- `count_nonzero_model_params()` — count active parameters after pruning
- `measure_keras_file_size()` — measure saved model size on disk

**Public API surface**
- `neat_optim.NEAT`, `NEATConfig`, `PlayerNEATConfig`
- `neat_optim.ArrayState`, `StepResult`, `StepMetrics`
- `neat_optim.engine.functional.neat_step` — framework-agnostic step function
- `neat_optim.engine.multiplayer.neat_player_step` — per-player step function
- `neat_optim.training.create_player_states`, `player_train_step`
- Lazy imports with clear error messages when optional dependencies are absent

**Native CPU extension**
- Optional C++ extension (`cpp/neat_core/`) via pybind11 and scikit-build-core
- In-place parameter update supporting `projection`, `cosine`, and `off` NCE modes
- Auto-fallback to reference engine when native core is unavailable or config is unsupported
- `native` config field: `"auto"` | `"never"` | `"force"`

**Tests**
- Unit tests: config validation, reference engine math, functional API, native parity, multiplayer engine
- Integration tests: Keras optimizer end-to-end, TensorFlow player training, model compaction
- Property tests: randomized invariant checking
- Regression tests: reproducible worked example from the math spec

**Benchmarks and examples**
- `benchmarks/run.py` — reproducible Keras MLP benchmark against SGD, Adam, AdamW
- `benchmarks/sweep_neat.py` — coarse hyperparameter sweep over NEAT config
- `benchmarks/compare_optimizers.py` — side-by-side optimizer comparison
- `examples/minimal_numpy.py` — pure NumPy usage
- `examples/keras_mlp_tf.py` — standard `model.compile` + `model.fit` usage
- `examples/keras_manual_apply.py` — custom training loop with `apply_gradients`
- `examples/player_aware_tensorflow.py` — per-example player-aware training

**Infrastructure**
- `pyproject.toml` with scikit-build-core, optional Keras and dev extras
- `py.typed` marker for PEP 561 compliance
- Ruff linting and formatting configuration
- Mypy strict type checking configuration
- pytest + pytest-cov configuration
- GitHub Actions CI: lint, type check, unit tests across Python 3.10–3.13
- GitHub Actions Keras integration workflow
- GitHub Actions wheel build and smoke test workflow
- cibuildwheel configuration for multi-platform distribution
- MkDocs documentation site with strict build mode
