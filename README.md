<div align="center">

# NEAT

> **Name note:** NEAT in this repository means Nash-Equilibrium Adaptive
> Training. It is unrelated to NeuroEvolution of Augmenting Topologies (NEAT;
> Stanley and Miikkulainen, 2002).

### Nash-Equilibrium Adaptive Training

NEAT, short for Nash-Equilibrium Adaptive Training, is a Keras-first optimizer
library for conflict-aware neural network optimization. It keeps the optimizer
math explicit and testable in a NumPy reference engine, while exposing a small
public API for real model training and an optional native CPU acceleration
path.

*A conflict-aware Keras optimizer with a versioned NumPy core*

<br>

[![CI](https://github.com/ItCodinTime/NEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/ItCodinTime/NEAT/actions/workflows/ci.yml)
[![Keras Integration](https://github.com/ItCodinTime/NEAT/actions/workflows/keras-integration.yml/badge.svg)](https://github.com/ItCodinTime/NEAT/actions/workflows/keras-integration.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0.dev0-orange)](CHANGELOG.md)

<br>

```
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" tensorflow
```

</div>

---

NEAT detects when the current gradient **conflicts** with the optimizer's
opponent signal and reduces the conflicting component before updating. The
correction is gated on a measured conflict ratio and clipped relative to the
gradient norm.

- opponent source selection: `momentum`, `previous_gradient`, or `gradient_ema`
- correction warmup via `correction_warmup_steps`
- conflict gating via `conflict_threshold`
- adaptive alpha via `adaptive_alpha`
- diagonal second-moment preconditioning via `adaptive_preconditioning`
- Lion-style sign updates via `update_mode="lion"`
- gradient centralization via `gradient_centralization`
- Nesterov-style momentum via `nesterov`
- Lookahead slow-weight synchronization via `lookahead_k`
- per-run optimizer diagnostics via `diagnostic_snapshot()`
  or epoch-level TensorBoard logging with `NEATDiagnosticsCallback`

The update math lives in a versioned NumPy reference engine, fully independent of Keras. The Keras optimizer is a thin adapter on top. An optional C++ CPU extension drops in behind the same semantics.

- Public training API: Keras optimizer subclass
- Reference core: NumPy
- Native acceleration: optional CPU-only C++ extension
- Tested platforms: Linux and macOS
- Supported Python versions: 3.10 to 3.13
- Supported Keras setup: install `keras` plus a backend runtime such as
  TensorFlow. The optimizer is written against Keras 3 APIs; TensorFlow is the
  currently exercised integration backend in this repo.
- Optional PyTorch adapter: `neat_optim.TorchNEAT`, used by the modern
  reproducible benchmark suite.

The v3 engineering work is additive. `nce_spec_v2` remains the default and its
behavior is frozen; v3 research options are not promoted without an explicit
author decision and full cross-engine parity coverage.

To select the additive specification, pass `spec_version="nce_spec_v3"`.
Default-off v3 controls cover parameter-local conflict granularity, an L2
trust-region correction bound, scheduled momentum/gradient-EMA opponent
ensembles, correction/preconditioner composition order, and deterministic
reference reductions. Existing constructors that omit `spec_version` continue
to use v2 behavior.

For large PyTorch models, `TorchNEAT(..., diagnostic_interval=10)` samples
diagnostics every ten steps. This reduces accelerator synchronization cost and
does not alter optimizer updates.

## Modern Benchmarks

The repository includes runnable comparisons for CIFAR-10/100 and ImageNet
folders with ResNet/ViT/DeiT, GLUE SST-2 and Alpaca small-LLM fine-tuning,
MuJoCo control, MNIST diffusion, and large-batch stability. Each run records
convergence, loss variance, wall time, environment metadata, and optimizer
diagnostics. See [the benchmark guide](benchmarks/torch_suite/README.md).
The dataset-free `python -m benchmarks.torch_suite.compare_all --quick` gate
runs five seeds across NEAT, AdamW, SGD momentum, Lion, PCGrad, and CAGrad in
their applicable synthetic settings.

## What's in 0.1.0.dev0

> **Pre-release.** Everything listed here is in the current development version (`0.1.0.dev0`); there is no tagged release yet. The public API is intentionally narrow — these are the pieces that have been validated end-to-end.

### Core algorithm

The NEAT update rule (`nce_spec_v1`) introduces a **Nash Conflict Elimination (NCE)** correction term:

```
conflict  =  relu( −cos(g, o) )          # positive only when gradient opposes momentum
proj      =  proj_o(g)                   # project gradient onto the opponent direction
nce       =  −α · conflict · proj        # subtract the conflicting component
u         =  g + nce                     # corrected gradient
m         =  β·m_prev + (1−β)·u          # standard momentum
θ_next    =  (1 − lr·wd)·θ  −  lr·m     # decoupled weight decay (default)
```

The correction is clipped: `‖nce‖ ≤ nce_clip_ratio · ‖g‖`. If there is no conflict, `nce = 0` and NEAT degrades to standard momentum SGD.

The full specification — including invariant proofs and a worked numerical example — is in [`docs/research/math-spec.md`](docs/research/math-spec.md).

### Shipped in this release

| Component | Description |
|---|---|
| `NEAT` Keras optimizer | Drop-in `model.compile()` optimizer, fully serializable via `get_config()` |
| NumPy reference engine | Canonical implementation used for correctness validation and framework-free use |
| Functional API | `neat_step(param, grad, state, config)` — works without Keras or TensorFlow |
| Player-aware mode | Treats each per-example gradient as a separate player in a custom TF training loop |
| Native CPU extension | Optional C++ extension (pybind11) with automatic fallback to the reference engine |
| Model compaction | `compact_dense_model()` and `search_compact_dense_model()` for sparse model search |
| Diagnostics API | `diagnostic_snapshot()` to measure correction activity during and after training |

### Configurable in this release

<details>
<summary><strong>Opponent signal</strong> — what counts as the conflict opponent</summary>

| Value | Behaviour |
|---|---|
| `"momentum"` (default) | Uses the accumulated momentum vector |
| `"gradient_ema"` | Uses an exponential moving average of raw gradients |
| `"previous_gradient"` | Uses the previous raw gradient |
| `"blended"` | Weighted mix of momentum and gradient EMA |

</details>

<details>
<summary><strong>Correction controls</strong> — when and how much to correct</summary>

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | `0.25` | Correction strength — `0` disables correction entirely |
| `nce_clip_ratio` | `1.0` | Max correction size as fraction of gradient norm |
| `correction_warmup_steps` | `0` | Steps before correction activates |
| `conflict_threshold` | `0.0` | Minimum conflict ratio to apply any correction |
| `nce_mode` | `"projection"` | `"projection"`, `"cosine"`, or `"off"` |

</details>

<details>
<summary><strong>Adaptive features</strong> — optional scaling and preconditioning</summary>

| Parameter | Default | Effect |
|---|---|---|
| `adaptive_correction` | `False` | Scale correction by opponent reliability and conflict EMA |
| `adaptive_correction_decay` | `0.9` | EMA decay for adaptive correction |
| `adaptive_correction_min_scale` | `1.0` | Lower clamp on adaptive scale |
| `adaptive_correction_max_scale` | `3.0` | Upper clamp on adaptive scale |
| `adaptive_preconditioning` | `False` | Adam-style second-moment preconditioner |
| `second_moment_beta` | `0.999` | Decay for second-moment EMA |
| `bias_correction` | `False` | Bias-correct momentum and second moment |

</details>

<details>
<summary><strong>Sparsity and weight decay</strong></summary>

| Parameter | Default | Effect |
|---|---|---|
| `weight_decay` | `0.0` | L2 weight decay coefficient |
| `decouple_weight_decay` | `True` | Apply decay before the gradient step (AdamW-style) |
| `sparsity_l1` | `0.0` | Soft L1 shrinkage after each step |
| `prune_threshold` | `0.0` | Hard-zero weights below this magnitude |

</details>

### Known limitations in 0.1.0.dev0

- Native C++ extension is CPU-only — no GPU kernel yet
- Player-aware mode requires TensorFlow as the Keras backend
- Published evidence currently consists of small Keras studies plus preliminary
  CPU PyTorch studies; no GPU-scale ImageNet, full foundation-model, or
  production RL claim is made

---

## Installation

> **Not on PyPI yet.** Install from GitHub as shown below. pip builds the native C++
> extension during install, so you need a C++ compiler (`xcode-select --install` on
> macOS, `build-essential` on Debian/Ubuntu).

**With TensorFlow** (Linux / macOS):
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" tensorflow
```

**With PyTorch** (any platform, including Apple Silicon):
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" torch
```

**With JAX**:
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" jax
```

**NumPy core only** (no framework required):
```bash
pip install "neat-optim @ git+https://github.com/ItCodinTime/NEAT.git"
```

**Local development**:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,keras]" && pip install tensorflow
```

---

## Quick Start

### Standard Keras training

```python
import keras
from neat_optim import NEAT

model = keras.Sequential(
    [
        keras.layers.Input((32,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10),
    ]
)

optimizer = NEAT(
    learning_rate=1e-3,
    alpha=0.25,
    beta=0.9,
    nce_mode="projection",
    opponent_source="gradient_ema",
    correction_warmup_steps=5,
    adaptive_alpha=True,
    adaptive_preconditioning=True,
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

# See how much the correction engaged
print(model.optimizer.diagnostic_snapshot())
```

<details>
<summary><strong>Custom training loop</strong></summary>

```python
import tensorflow as tf
import keras
from neat_optim import NEAT

optimizer = NEAT(learning_rate=1e-3, alpha=0.25)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for x_batch, y_batch in dataset:
    loss = train_step(x_batch, y_batch)
```

</details>

<details>
<summary><strong>Framework-free NumPy engine</strong></summary>

```python
import numpy as np
from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState

param  = np.array([1.0, -2.0, 0.5], dtype=np.float32)
grad   = np.array([0.5, -0.25, 0.1], dtype=np.float32)
state  = ArrayState.zeros_like(param)
config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9)

for _ in range(10):
    result = neat_step(param, grad, state, config)
    param  = result.param
    state  = result.state

print(result.metrics)
# StepMetrics(grad_norm=0.568, update_norm=0.370, nce_norm=0.0, conflict_ratio=0.0, ...)
# conflict_ratio and nce_norm are 0.0 here: a constant gradient never conflicts
# with its own momentum, so the correction stays off.
```

</details>

<details>
<summary><strong>Per-example player-aware training</strong></summary>

```python
import keras
from neat_optim import PlayerNEATConfig
from neat_optim.training import create_player_states, player_train_step

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
])
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

states = create_player_states(model)
config = PlayerNEATConfig(
    learning_rate=1e-2,
    alpha=0.25,
    sparsity_l1=1e-4,
    prune_threshold=1e-3,
)

for x_batch, y_batch in dataset:
    result = player_train_step(model, x_batch, y_batch, loss_fn, states, config)
    states = result.states
    print(f"loss={result.loss:.4f}")
```

</details>

---

## Public API

- `neat_optim.NEAT`
- `neat_optim.NEATDiagnosticsCallback`
- `neat_optim.NEATConfig`
- `neat_optim.PlayerNEATConfig`
- `neat_optim.ArrayState`
- `neat_optim.engine.functional.neat_step`
- `neat_optim.engine.multiplayer.neat_player_step`
- `neat_optim.training.player_train_step`

## Diagnostics

Call `diagnostic_snapshot()` after training to check whether the correction engaged:

```python
snap = model.optimizer.diagnostic_snapshot()
# {
#   "mean_conflict_ratio":       0.021,   # average conflict detected per step
#   "mean_correction_ratio":     0.008,   # correction size / gradient size
#   "mean_update_alignment":     0.997,   # cosine similarity of corrected vs raw gradient
#   "mean_opponent_norm":        0.043,   # magnitude of the opponent signal
#   "correction_active_fraction": 0.74,   # fraction of steps where correction fired
# }
```

> **Interpreting results:** If `mean_conflict_ratio < 0.01` and `correction_active_fraction < 0.1`, the correction is essentially inactive on this task. NEAT will behave like standard momentum SGD and tuning `alpha` won't help. Tasks with noisy or multi-objective gradients tend to show much higher conflict ratios.

Optional research modes extend the same base update without changing defaults:

- `adaptive_alpha=True` makes the correction strength respond to conflict,
  gradient-noise, and alignment trends.
- `adaptive_preconditioning=True` adds a cheap diagonal second-moment
  preconditioner, similar in spirit to Fisher/Hessian-diagonal scaling.
- `update_mode="lion"` applies a sign-based final update after the NEAT
  correction.
- `gradient_centralization=True`, `nesterov=True`, and `lookahead_k > 0` enable
  standard training stabilizers that are useful benchmark ablations.

This is Nash-inspired because the optimizer reacts to directional conflict
between the current gradient and an opponent proxy. The exact behavior is
defined in [`docs/research/math-spec.md`](docs/research/math-spec.md).

Reset before each epoch to get per-epoch measurements:

```python
for epoch in range(epochs):
    model.optimizer.reset_diagnostics()
    model.fit(x_train, y_train, epochs=1, verbose=0)
    snap = model.optimizer.diagnostic_snapshot()
    print(f"Epoch {epoch+1}: conflict={snap['mean_conflict_ratio']:.4f}, "
          f"active={snap['correction_active_fraction']:.2%}")
```

---

## Benchmarks

The repository includes both lightweight local checks and larger benchmark
harnesses:

- lint checks via `ruff`
- package build via `python -m build`
- docs build via `mkdocs build --strict`
- Keras integration tests
- reference/native parity tests
- a real Keras MLP benchmark against SGD, Adam, and AdamW
- a real Keras CNN benchmark on MNIST and Fashion-MNIST
- runnable benchmark harnesses for CIFAR-10 and GLUE SST-2
- benchmark diagnostics and sweep tooling for NEAT-specific ablations

Evaluated on the sklearn digits dataset, two-layer MLP, 20 epochs, 3 seeds:

| Optimizer | Test Accuracy | Notes |
|---|---|---|
| SGD + momentum | **97.04%** | lr=0.01, momentum=0.9 |
| Adam | **96.85%** | default hyperparameters |
| AdamW | **96.85%** | weight_decay=1e-4 |
| NEAT | 94.72% | α=0.25, β=0.9, gradient_ema opponent |

On the current 20-epoch Keras digits benchmark, NEAT reaches `0.9472` mean
test accuracy and trails SGD/Adam baselines. The attached diagnostics show why
that is plausible: the mean correction ratio is only `0.00385` and the mean
gradient/update alignment is `0.99991`, so NEAT is behaving very close to its
base update on this well-aligned task.

On a stronger short-transfer benchmark with a small CNN on `MNIST` and
`Fashion-MNIST` over 3 seeds and 2 epochs, adaptive NEAT now reaches the best
mean test accuracy on both datasets: `0.9861` vs `0.9856` for Adam on MNIST,
and `0.8786` vs `0.8725` for Adam on Fashion-MNIST. That is better evidence
than the digits-only benchmark, but it is still not a substitute for broader
GPU-side benchmarks such as ImageNet or GLUE.

To reproduce the benchmark:

```bash
python benchmarks/run.py          # reproduce the table above
python benchmarks/sweep_neat.py   # hyperparameter sweep
python benchmarks/compare_optimizers.py  # side-by-side comparison
```

Full methodology and results: [`docs/research/benchmarks.md`](docs/research/benchmarks.md).

---

## Repository Layout

```
neat-optim/
├── src/neat_optim/
│   ├── engine/
│   │   ├── reference.py      # canonical NumPy implementation (nce_spec_v1)
│   │   ├── functional.py     # public neat_step() API with native fallback
│   │   ├── multiplayer.py    # per-player stepping
│   │   └── native.py         # native extension bridge
│   ├── training/
│   │   └── tensorflow_players.py  # TF player-aware helpers
│   ├── config.py             # NEATConfig, PlayerNEATConfig
│   ├── state.py              # ArrayState, StepResult, StepMetrics
│   ├── keras_optimizer.py    # Keras adapter
│   └── compaction.py         # dense model pruning utilities
├── cpp/neat_core/            # optional C++ extension (pybind11)
├── tests/
│   ├── unit/                 # algorithm correctness, config validation
│   ├── integration/          # Keras optimizer, player training, compaction
│   ├── property/             # randomized invariant checks
│   └── regression/           # reproducibility of documented examples
├── benchmarks/               # reproducible benchmark entrypoints
├── examples/                 # minimal runnable scripts
└── docs/
    ├── research/             # math spec, algorithm, player-aware, benchmarks
    └── contributing/         # architecture, development, release workflow
```

---

To run the short standard vision benchmark:

```bash
python benchmarks/vision_adaptive_neat_vs_baselines.py
```

To run the CIFAR-10 benchmark harness:

```bash
python benchmarks/cifar10_adaptive_neat_vs_baselines.py
```

To run the GLUE SST-2 benchmark harness:

```bash
python benchmarks/glue_sst2_adaptive_neat_vs_baselines.py
```

The CIFAR-10 and GLUE SST-2 harnesses are included so the repo can scale to
stronger benchmark environments. They are runnable here, but full credible
ImageNet- or broad-GLUE-style evidence still requires a stronger machine than
this local CPU-only setup.

## Development

```bash
ruff check .                                           # lint
ruff format .                                          # format
mypy src/neat_optim                                    # type check
pytest tests/unit                                      # fast, no Keras required
pytest                                                 # full suite
pytest --cov=neat_optim --cov-report=term-missing      # with coverage
python -m build                                        # build wheel + sdist
mkdocs build --strict                                  # docs build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, PR standards, and how to report bugs.

---

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Installation and first usage |
| [API Reference](docs/api.md) | Full public API |
| [Math Spec](docs/research/math-spec.md) | Versioned update rule (`nce_spec_v1`) |
| [Algorithm](docs/research/algorithm.md) | Motivation and design |
| [Player-aware mode](docs/research/player-aware.md) | Per-example gradient stepping |
| [Benchmarks](docs/research/benchmarks.md) | Reproducible results |
| [Contributing](CONTRIBUTING.md) | Development workflow |
| [Changelog](CHANGELOG.md) | Release history |

---

## License

[Apache 2.0](LICENSE) — includes an explicit patent grant, which is the right default for optimization infrastructure.
