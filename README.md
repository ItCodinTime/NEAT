<div align="center">

# NEAT

### Conflict-aware optimization for Keras.

**Nash-Equilibrium Adaptive Training.** NEAT is a drop-in Keras 3 optimizer. It detects
when gradients fight each other and applies a bounded correction before each step.
It runs on TensorFlow, JAX and PyTorch.

<br>

[![CI](https://github.com/ItCodinTime/NEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/ItCodinTime/NEAT/actions/workflows/ci.yml)
[![Keras Integration](https://github.com/ItCodinTime/NEAT/actions/workflows/keras-integration.yml/badge.svg)](https://github.com/ItCodinTime/NEAT/actions/workflows/keras-integration.yml)
[![Keras 3](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)](#keras-3-every-backend)
[![Backends](https://img.shields.io/badge/backends-TensorFlow%20|%20JAX%20|%20PyTorch-555)](#keras-3-every-backend)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0.dev0-orange)](CHANGELOG.md)

[**Documentation**][docs] · [Installation][docs-install] · [Keras guide][docs-keras] · [API][docs-api] · [Benchmarks][docs-benchmarks]

</div>

```python
import keras
from neat_optim import NEAT

optimizer = NEAT(
    learning_rate=1e-3,
    alpha=0.25,
    nce_mode="projection",
    opponent_source="gradient_ema",
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=5)

print(optimizer.diagnostic_snapshot())
```

> [!NOTE]
> **Name note:** NEAT here means *Nash-Equilibrium Adaptive Training*. It is unrelated to
> NeuroEvolution of Augmenting Topologies (Stanley and Miikkulainen, 2002). It is also not
> neural architecture search, and not a full game-theory solver.

---

## Contents

- [Why NEAT](#why-neat)
- [Keras 3, every backend](#keras-3-every-backend)
- [Built for the Google DeepMind stack](#built-for-the-google-deepmind-stack)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Choose a mode](#choose-a-mode)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
- [Public API](#public-api)
- [Status and limitations](#status-and-limitations)
- [Development](#development)
- [Documentation](#documentation)

---

## Why NEAT

Standard optimizers apply every gradient as-is, even when the current gradient points against
the direction training has been moving. NEAT measures that **directional conflict** between the
gradient and an *opponent* signal (momentum, a gradient EMA, or the previous gradient). When it
finds conflict, it removes the conflicting component before the momentum update. The correction is:

- **gated**: no conflict means no correction, and NEAT behaves like momentum SGD
- **bounded**: clipped to `nce_clip_ratio · ‖g‖`, so it can never dominate the step
- **observable**: every run reports how often and how strongly the correction fired

The update math lives in a **versioned NumPy reference engine**, fully independent of Keras.
The Keras optimizer is a thin adapter on top of it, and an optional C++ CPU extension drops in
behind the same semantics. Parity tests keep all three in agreement.

| | |
|---|---|
| **Keras drop-in** | Pass `NEAT` to `model.compile(...)`. Serializable via `get_config()`. |
| **Reference engine** | Deterministic NumPy `neat_step(...)` for debugging and framework-free validation. |
| **Player-aware mode** | Treat each example, task, or objective head as its own gradient contributor. |
| **Diagnostics** | `diagnostic_snapshot()` and `NEATDiagnosticsCallback` (Keras `History` + TensorBoard). |
| **PyTorch adapter** | `neat_optim.TorchNEAT` (install the `[torch]` extra) for the modern PyTorch benchmark suite. |

---

## Keras 3, every backend

NEAT is written against Keras 3 optimizer APIs, so the same `model.compile(optimizer=NEAT(...))`
code runs on any Keras backend. Pick one with `KERAS_BACKEND`:

| Backend | Install | Status |
|---|---|---|
| **TensorFlow** | `pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" tensorflow` | Exercised in CI (Keras integration workflow) |
| **JAX** | `pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" "jax[cpu]"` | Smoke-tested locally: `fit()` + diagnostics callback |
| **PyTorch** | `pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" torch` | Smoke-tested locally: `fit()` + diagnostics callback |

```bash
KERAS_BACKEND=jax python train.py
```

> Player-aware training (`player_train_step`) uses `tf.GradientTape`, so it requires the
> TensorFlow backend.

The Keras integration goes beyond `compile()`:

- **Callbacks**: `NEATDiagnosticsCallback` adds `neat_*` metrics to `History`, writes TensorBoard
  scalars, and can export a JSON or HTML training report.
- **Custom loops**: `optimizer.apply_gradients(...)` works inside `tf.function` training steps.
- **Serialization**: full `get_config()` round-trip, so compiled models save and load normally.

---

## Built for the Google DeepMind stack

NEAT targets the tools that Google DeepMind's open ecosystem is built on:

| | What NEAT ships |
|---|---|
| **JAX** | Runs as a Keras 3 optimizer on the JAX backend; see [Keras 3, every backend](#keras-3-every-backend). |
| **Gemma** | The LLM fine-tuning harness supports `google/gemma-2b` (LoRA, AMP, gradient accumulation) on Alpaca and GLUE SST-2. |
| **MuJoCo** | The reinforcement-learning harness trains SAC policies on HalfCheetah-v5, Hopper-v5 and Walker2d-v5 with NEAT or AdamW. |

```bash
# Gemma-2B LoRA fine-tuning on Alpaca (requires Hugging Face auth for the Gemma weights)
python -m benchmarks.torch_suite.language_model \
    --task alpaca --model google/gemma-2b \
    --optimizer neat --lora --epochs 1 --lr 2e-5 \
    --batch-size 4 --gradient-accumulation 8 --amp
```

These are runnable harnesses. Gemma-scale and MuJoCo results have not been published yet; see
[Status and limitations](#status-and-limitations). Setup details are in the
[PyTorch benchmark guide](benchmarks/torch_suite/README.md).

<sub>NEAT is an independent open-source project. It is not affiliated with or endorsed by
Google or Google DeepMind. Gemma, JAX and MuJoCo are referenced only as tools NEAT works with.</sub>

---

## Installation

> [!IMPORTANT]
> **Not on PyPI yet.** Install from GitHub as shown below. pip builds the native C++
> extension during install, so you need a C++ compiler (`xcode-select --install` on
> macOS, `build-essential` on Debian/Ubuntu).

**Keras + TensorFlow**
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" tensorflow
```

**Keras + JAX**
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" "jax[cpu]"
```

**Keras + PyTorch** (any platform, including Apple Silicon)
```bash
pip install "neat-optim[keras] @ git+https://github.com/ItCodinTime/NEAT.git" torch
```

**NumPy core only** (no framework required)
```bash
pip install "neat-optim @ git+https://github.com/ItCodinTime/NEAT.git"
```

**Local development**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,keras]" && pip install tensorflow
```

Requirements: Python 3.10 to 3.13, `keras>=3.13,<4`, Linux or macOS.
→ Full guide: [Installation][docs-install]

---

## Quick start

### Keras training

```python
import keras
from neat_optim import NEAT

model = keras.Sequential([
    keras.layers.Input((32,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10),
])

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

### Diagnostics in TensorBoard

```python
from neat_optim import NEATDiagnosticsCallback

history = model.fit(
    x_train, y_train,
    callbacks=[
        NEATDiagnosticsCallback(
            log_dir="runs/neat",                          # TensorBoard scalars (pip install tensorboard)
            report_path="runs/neat/training-report.html", # optional JSON/HTML report
            reset_each_epoch=True,
        ),
    ],
)
print(history.history["neat_mean_conflict_ratio"])
```

<details>
<summary><strong>Custom TensorFlow training loop</strong></summary>

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
<summary><strong>Per-example player-aware training (TensorFlow backend)</strong></summary>

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

→ Full guides: [Keras][docs-keras] · [Player-aware NEAT][docs-player]

---

## Choose a mode

| If you want to… | Use | Guide |
|---|---|---|
| Train a regular Keras model | `NEAT` in `model.compile(...)` | [Keras][docs-keras] |
| See whether the correction is doing anything | `diagnostic_snapshot()` / `NEATDiagnosticsCallback` | [Diagnostics](#diagnostics) |
| Debug or experiment with the update rule | `neat_step(...)` (NumPy, deterministic) | [API][docs-api] |
| Give each example or task its own gradient | `player_train_step(...)` | [Player-aware][docs-player] |
| Train PyTorch models directly | `TorchNEAT` | [Benchmark guide](benchmarks/torch_suite/README.md) |

### Diagnostics

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

> [!TIP]
> If `mean_conflict_ratio < 0.01` and `correction_active_fraction < 0.1`, the correction is
> essentially inactive on this task. NEAT will behave like momentum SGD and tuning `alpha`
> won't help. Tasks with noisy or multi-objective gradients tend to show much higher
> conflict ratios.

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

## How it works

The current spec is `nce_spec_v2`. Each step adds a **Nash Conflict Elimination (NCE)**
correction:

```text
o         =  opponent_proxy(g, m_prev)   # momentum, gradient EMA, or previous gradient
conflict  =  relu( −cos(g, o) )          # positive only when the gradient opposes the opponent
proj      =  proj_o(g)                   # project the gradient onto the opponent direction
nce       =  −α · conflict · proj        # subtract the conflicting component
u         =  g + nce                     # corrected gradient
m         =  β·m_prev + (1−β)·u          # standard momentum
θ_next    =  (1 − lr·wd)·θ  −  lr·m      # decoupled weight decay (default)
```

The correction is clipped: `‖nce‖ ≤ nce_clip_ratio · ‖g‖`. If there is no conflict, `nce = 0`
and NEAT degrades to standard momentum SGD. With `adaptive_alpha=True`, `α` responds to
conflict, gradient-noise, and alignment trends.

It is *Nash-inspired* because the optimizer reacts to directional conflict between the
current gradient and an opponent proxy. The full specification, including invariant proofs and
a worked numerical example, is in [`docs/research/math-spec.md`](docs/research/math-spec.md).
→ [Algorithm][docs-algorithm]

<details>
<summary><strong>Spec versions (<code>nce_spec_v2</code> default, <code>nce_spec_v3</code> opt-in)</strong></summary>

`nce_spec_v2` is the default and its behavior is frozen. The v3 work is additive: v3 research
options are not promoted without an explicit author decision and full cross-engine parity
coverage.

To select the additive specification, pass `spec_version="nce_spec_v3"`. Default-off v3
controls cover parameter-local conflict granularity, an L2 trust-region correction bound,
scheduled momentum/gradient-EMA opponent ensembles, correction/preconditioner composition
order, and deterministic reference reductions. Existing constructors that omit `spec_version`
continue to use v2 behavior.

</details>

---

## Configuration

<details>
<summary><strong>Opponent signal</strong>: what counts as the conflict opponent</summary>

| Value | Behaviour |
|---|---|
| `"momentum"` (default) | Uses the accumulated momentum vector |
| `"gradient_ema"` | Uses an exponential moving average of raw gradients |
| `"previous_gradient"` | Uses the previous raw gradient |
| `"blended"` | Weighted mix of momentum and gradient EMA |

</details>

<details>
<summary><strong>Correction controls</strong>: when and how much to correct</summary>

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | `0.25` | Correction strength (`0` disables correction entirely) |
| `nce_clip_ratio` | `1.0` | Max correction size as fraction of gradient norm |
| `correction_warmup_steps` | `0` | Steps before correction activates |
| `conflict_threshold` | `0.0` | Minimum conflict ratio to apply any correction |
| `nce_mode` | `"projection"` | `"projection"`, `"cosine"`, or `"off"` |

</details>

<details>
<summary><strong>Adaptive features</strong>: optional scaling and preconditioning</summary>

| Parameter | Default | Effect |
|---|---|---|
| `adaptive_alpha` | `False` | Correction strength responds to conflict, noise, and alignment trends |
| `adaptive_correction` | `False` | Scale correction by opponent reliability and conflict EMA |
| `adaptive_correction_decay` | `0.9` | EMA decay for adaptive correction |
| `adaptive_correction_min_scale` | `1.0` | Lower clamp on adaptive scale |
| `adaptive_correction_max_scale` | `3.0` | Upper clamp on adaptive scale |
| `adaptive_preconditioning` | `False` | Diagonal second-moment preconditioner (Fisher/Hessian-diagonal in spirit) |
| `second_moment_beta` | `0.999` | Decay for second-moment EMA |
| `bias_correction` | `False` | Bias-correct momentum and second moment |

</details>

<details>
<summary><strong>Training stabilizers</strong>: useful benchmark ablations</summary>

| Parameter | Effect |
|---|---|
| `update_mode="lion"` | Lion-style sign update after the NEAT correction |
| `gradient_centralization=True` | Center matrix-like gradients before the update |
| `nesterov=True` | Nesterov-style momentum |
| `lookahead_k > 0` | Lookahead slow-weight synchronization every `k` steps |

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

→ Full reference: [Configuration][docs-config]

---

## Benchmarks

> [!WARNING]
> These are early results. Read them as evidence, not as a claim that NEAT beats every
> other optimizer.

**Short-transfer Keras CNN** (MNIST and Fashion-MNIST, 3 seeds, 2 epochs). Adaptive NEAT
reaches the best mean test accuracy on both datasets:

| Dataset | Adaptive NEAT | Adam |
|---|---|---|
| MNIST | **0.9861** | 0.9856 |
| Fashion-MNIST | **0.8786** | 0.8725 |

**Keras MLP** (sklearn digits, two-layer MLP, 20 epochs, 3 seeds). NEAT trails the baselines:

| Optimizer | Test accuracy | Notes |
|---|---|---|
| SGD + momentum | **97.04%** | lr=0.01, momentum=0.9 |
| Adam | 96.85% | default hyperparameters |
| AdamW | 96.85% | weight_decay=1e-4 |
| NEAT | 94.72% | α=0.25, β=0.9, gradient_ema opponent |

The diagnostics explain why: on this well-aligned task the mean correction ratio is only
`0.00385` and gradient/update alignment is `0.99991`, so NEAT behaves very close to its base
update.

### Modern benchmark suite (PyTorch)

Runnable comparisons cover image classification (ResNet-18/34 on CIFAR-10/100 and
ImageFolder), vision transformers (ViT-S/16, DeiT-S/16), LLM fine-tuning (TinyLlama-1.1B,
Phi-2, Gemma-2B on GLUE SST-2 and Alpaca), MuJoCo control (SAC), MNIST diffusion (UNet-2D
DDPM), and large-batch stability. Each run records convergence, loss variance, wall time,
environment metadata, and optimizer diagnostics. The dataset-free gate runs five seeds across
NEAT, AdamW, SGD momentum, Lion, PCGrad, and CAGrad:

```bash
python -m benchmarks.torch_suite.compare_all --quick
```

<details>
<summary><strong>Reproduce the Keras benchmarks</strong></summary>

```bash
python benchmarks/run.py                                # MLP table above
python benchmarks/vision_adaptive_neat_vs_baselines.py  # MNIST / Fashion-MNIST CNN
python benchmarks/cifar10_adaptive_neat_vs_baselines.py # CIFAR-10 harness
python benchmarks/glue_sst2_adaptive_neat_vs_baselines.py # GLUE SST-2 harness
python benchmarks/sweep_neat.py                         # hyperparameter sweep
python benchmarks/compare_optimizers.py                 # side-by-side comparison
```

</details>

For large PyTorch models, `TorchNEAT(..., diagnostic_interval=10)` samples diagnostics every
ten steps. This reduces accelerator synchronization cost and does not change the optimizer
updates.

→ Methodology and full results: [Benchmarks][docs-benchmarks] ·
[`docs/research/benchmarks.md`](docs/research/benchmarks.md) ·
[PyTorch suite guide](benchmarks/torch_suite/README.md)

---

## Public API

| Symbol | Purpose |
|---|---|
| `neat_optim.NEAT` | Keras optimizer for `model.compile` |
| `neat_optim.NEATDiagnosticsCallback` | Epoch-level diagnostics in `History` / TensorBoard |
| `neat_optim.NEATConfig` | Reference engine configuration |
| `neat_optim.PlayerNEATConfig` | Configuration for per-player gradient updates |
| `neat_optim.ArrayState` | Per-parameter reference engine state |
| `neat_optim.engine.functional.neat_step` | Framework-agnostic one-step update |
| `neat_optim.engine.multiplayer.neat_player_step` | Player-aware one-step update with conflict/sparsity metrics |
| `neat_optim.training.player_train_step` | TensorFlow helper for custom loops with per-example losses |
| `neat_optim.TorchNEAT` | Optional PyTorch optimizer adapter (requires `neat-optim[torch]`) |

→ Full reference: [API][docs-api] · [`docs/api.md`](docs/api.md)

---

## Status and limitations

> [!NOTE]
> **Pre-release (`0.1.0.dev0`).** There is no tagged release yet. The public API is
> intentionally narrow: these are the pieces that have been validated end-to-end.

| Component | Description |
|---|---|
| `NEAT` Keras optimizer | Drop-in `model.compile()` optimizer, fully serializable via `get_config()` |
| NumPy reference engine | Canonical implementation used for correctness validation and framework-free use |
| Functional API | `neat_step(param, grad, state, config)`, which works without Keras or TensorFlow |
| Player-aware mode | Treats each per-example gradient as a separate player in a custom TF training loop |
| Native CPU extension | Optional C++ extension (pybind11) with automatic fallback to the reference engine |
| Model compaction | `compact_dense_model()` and `search_compact_dense_model()` for sparse model search |
| Diagnostics API | `diagnostic_snapshot()` to measure correction activity during and after training |

**Known limitations**

- The native C++ extension is CPU-only, with no GPU kernel yet.
- Player-aware mode requires TensorFlow as the Keras backend.
- CI exercises the TensorFlow backend. JAX and PyTorch backends are smoke-tested, not yet in CI.
- Published evidence currently consists of small Keras studies plus preliminary CPU PyTorch
  studies. No GPU-scale ImageNet, full foundation-model, or production RL claim is made.

---

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

<details>
<summary><strong>Repository layout</strong></summary>

```
neat-optim/
├── src/neat_optim/
│   ├── engine/
│   │   ├── reference.py      # canonical NumPy implementation
│   │   ├── functional.py     # public neat_step() API with native fallback
│   │   ├── multiplayer.py    # per-player stepping
│   │   └── native.py         # native extension bridge
│   ├── training/
│   │   └── tensorflow_players.py  # TF player-aware helpers
│   ├── config.py             # NEATConfig, PlayerNEATConfig
│   ├── state.py              # ArrayState, StepResult, StepMetrics
│   ├── keras_optimizer.py    # Keras adapter
│   ├── callbacks.py          # NEATDiagnosticsCallback
│   └── compaction.py         # dense model pruning utilities
├── cpp/neat_core/            # optional C++ extension (pybind11)
├── tests/
│   ├── unit/                 # algorithm correctness, config validation
│   ├── integration/          # Keras optimizer, player training, compaction
│   ├── property/             # randomized invariant checks
│   └── regression/           # reproducibility of documented examples
├── benchmarks/               # Keras benchmarks + torch_suite/ modern PyTorch suite
├── examples/                 # minimal runnable scripts
└── docs/
    ├── research/             # math spec, algorithm, player-aware, benchmarks
    └── contributing/         # architecture, development, release workflow
```

</details>

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, PR standards, and how to report
bugs. → [Development][docs-dev]

---

## Documentation

The documentation site at **[NEAT Docs][docs]** covers everything in this README in more depth:

| Docs site | In this repo | |
|---|---|---|
| [Introduction][docs] | [README](README.md) | What NEAT is and isn't |
| [Installation][docs-install] | [docs/quickstart.md](docs/quickstart.md) | Install options and requirements |
| [Keras][docs-keras] | [examples/](examples) | `model.compile` usage and diagnostics |
| [Choose a mode][docs-modes] | | Picking the right entry point |
| [Player-aware NEAT][docs-player] | [docs/research/player-aware.md](docs/research/player-aware.md) | Per-example gradient stepping |
| [API reference][docs-api] | [docs/api.md](docs/api.md) | Full public API |
| [Configuration][docs-config] | | Optimizer arguments |
| [Algorithm][docs-algorithm] | [docs/research/math-spec.md](docs/research/math-spec.md) | Versioned update rule |
| [Benchmarks][docs-benchmarks] | [docs/research/benchmarks.md](docs/research/benchmarks.md) | Reproducible results |
| [Development][docs-dev] | [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow |
| [Resources][docs-resources] | [CHANGELOG.md](CHANGELOG.md) | Policies, changelog, contact |

Questions: open a [GitHub issue](https://github.com/ItCodinTime/NEAT/issues) or email
[gouthamkrishnaronanki@gmail.com](mailto:gouthamkrishnaronanki@gmail.com).

---

## License

[Apache 2.0](LICENSE), which includes an explicit patent grant: the right default for
optimization infrastructure.

<!-- Docs site links: set DOCS_URL below to the live docs domain (e.g. the Vercel production URL). -->
[docs]: https://DOCS_URL/docs
[docs-install]: https://DOCS_URL/docs/installation
[docs-keras]: https://DOCS_URL/docs/keras
[docs-modes]: https://DOCS_URL/docs/modes
[docs-player]: https://DOCS_URL/docs/player-aware
[docs-api]: https://DOCS_URL/docs/api
[docs-config]: https://DOCS_URL/docs/configuration
[docs-algorithm]: https://DOCS_URL/docs/algorithm
[docs-benchmarks]: https://DOCS_URL/docs/benchmarks
[docs-dev]: https://DOCS_URL/docs/development
[docs-resources]: https://DOCS_URL/docs/resources
