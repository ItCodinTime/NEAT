# NEAT

NEAT, short for Nash-Equilibrium Adaptive Training, is a Keras-first optimizer
library for conflict-aware neural network optimization. It keeps the optimizer
math explicit and testable in a NumPy reference engine, while exposing a small
public API for real model training and an optional native CPU acceleration
path.

## What NEAT Is

- A Keras optimizer you can use in `model.compile(...)`
- A reference NumPy engine for deterministic algorithm validation
- A player-aware engine that can treat each example or task gradient as a
  player in a custom training loop
- A research-friendly implementation of a Nash-inspired correction term
- A small package with a narrow public surface and explicit math spec

## What NEAT Is Not

- A neural architecture search library
- A framework for building models for you
- A full Nash-equilibrium solver for general multi-agent games
- A claim of universal superiority over Adam or SGD

The first release uses the previous momentum vector as an opponent proxy and
applies a conflict correction when the current gradient moves against that
signal.

The repository also includes a separate player-aware mode that forms an
opponent proxy from other examples in the batch and can add sparsity or hard
pruning pressure for lighter models.

The standard optimizer now also exposes research knobs for:

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

## Current Status

- Public training API: Keras optimizer subclass
- Reference core: NumPy
- Native acceleration: optional CPU-only C++ extension
- Tested platforms: Linux and macOS
- Supported Python versions: 3.10 to 3.13
- Supported Keras setup: install `keras` plus a backend runtime such as
  TensorFlow. The optimizer is written against Keras 3 APIs; TensorFlow is the
  currently exercised integration backend in this repo.

## Why This Repository Exists

Most optimizer repos mix together framework glue, experimental math, and
performance code. NEAT keeps those concerns separate:

- the update rule is specified and tested independently of Keras
- the Keras adapter stays small and serialization-friendly
- the optional native core is constrained behind the same reference semantics

That structure makes the repo easier to review, benchmark, and evolve without
hiding behavior in framework-specific internals.

## Installation

Install the core package:

```bash
pip install neat-optim
```

Install Keras integration:

```bash
pip install "neat-optim[keras]" tensorflow
```

Install a local development environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,keras]"
```

For core-only validation on Python 3.13:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

### Train a Keras Model

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

history = model.fit(x_train, y_train, epochs=5, verbose=0)
print(optimizer.diagnostic_snapshot())
```

### Use the Reference Engine Directly

```python
import numpy as np
from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState

param = np.array([1.0, -2.0], dtype=np.float32)
grad = np.array([0.5, -0.25], dtype=np.float32)
state = ArrayState.zeros_like(param)
config = NEATConfig(
    learning_rate=0.1,
    alpha=0.25,
    beta=0.9,
    nce_mode="projection",
)

result = neat_step(param, grad, state, config)
print(result.param)
print(result.state.momentum)
print(result.metrics)
```

### Treat Each Example as a Player

```python
import keras
from neat_optim import PlayerNEATConfig
from neat_optim.training import create_player_states, player_train_step

model = keras.Sequential(
    [
        keras.layers.Input((32,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none",
)
states = create_player_states(model)
config = PlayerNEATConfig(
    learning_rate=1e-2,
    alpha=0.25,
    beta=0.9,
    sparsity_l1=1e-4,
    prune_threshold=1e-3,
)

result = player_train_step(model, x_batch, y_batch, loss_fn, states, config)
states = result.states
```

## Public API

The first release intentionally keeps the public API narrow:

- `neat_optim.NEAT`
- `neat_optim.NEATDiagnosticsCallback`
- `neat_optim.NEATConfig`
- `neat_optim.PlayerNEATConfig`
- `neat_optim.ArrayState`
- `neat_optim.engine.functional.neat_step`
- `neat_optim.engine.multiplayer.neat_player_step`
- `neat_optim.training.player_train_step`

The Keras optimizer also exposes `diagnostic_snapshot()` and
`reset_diagnostics()` for benchmark and experiment code.

## How the Update Works

For parameter tensor `theta_t`, gradient `g_t`, opponent proxy `o_t`, and
correction scale `alpha`:

```text
conflict_t = relu(-cos(g_t, o_t))
proj_t     = proj_{o_t}(g_t)
nce_t      = -alpha * conflict_t * proj_t
u_t        = g_t + nce_t
m_t        = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1  = (1 - lr * wd) * theta_t - lr * m_t
```

The default `o_t` is the previous momentum vector, but the standard optimizer
can also use the previous raw gradient or an exponential moving average of
gradients. The correction can be delayed with a warmup or suppressed unless
the measured conflict clears a threshold.

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

For explicit player-aware stepping, the opponent proxy is built from other
players in the batch. That mode is documented in
[`docs/research/player-aware.md`](docs/research/player-aware.md).

## Lightweight Models

NEAT can now apply lightweight-model pressure through two optional controls:

- `sparsity_l1`: soft-threshold shrinkage after each step
- `prune_threshold`: hard pruning of small-magnitude weights to zero

These settings encourage sparse parameters. They do not redesign the model
architecture or automatically remove layers.

## Repository Structure

```text
src/neat_optim/      Package source
cpp/neat_core/       Optional native CPU extension
tests/               Unit, regression, property, and integration tests
benchmarks/          Reproducible benchmark entrypoints
examples/            Minimal runnable examples
docs/                User, research, and contributor documentation
```

## Validation Snapshot

At the time of the current repository hardening pass, the project validates
cleanly with:

- lint checks via `ruff`
- package build via `python -m build`
- docs build via `mkdocs build --strict`
- Keras integration tests
- reference/native parity tests
- a real Keras MLP benchmark against SGD, Adam, and AdamW
- a real Keras CNN benchmark on MNIST and Fashion-MNIST
- runnable benchmark harnesses for CIFAR-10 and GLUE SST-2
- benchmark diagnostics and sweep tooling for NEAT-specific ablations

In a small real supervised-learning experiment on the `sklearn` digits dataset,
the reference engine successfully trained a two-layer MLP to `0.9194` test
accuracy after 10 epochs. That result is useful as a sanity check, not as a
benchmark claim.

On the current 20-epoch Keras digits benchmark, NEAT reaches `0.9472` mean
test accuracy and trails SGD/Adam baselines. The attached diagnostics show why
that is plausible: the mean correction ratio is only `0.00385` and the mean
gradient/update alignment is `0.99991`, so NEAT is behaving very close to its
base update on this task.

On a stronger short-transfer benchmark with a small CNN on `MNIST` and
`Fashion-MNIST` over 3 seeds and 2 epochs, adaptive NEAT now reaches the best
mean test accuracy on both datasets: `0.9861` vs `0.9856` for Adam on MNIST,
and `0.8786` vs `0.8725` for Adam on Fashion-MNIST. That is better evidence
than the digits-only benchmark, but it is still not a substitute for broader
GPU-side benchmarks such as ImageNet or GLUE.

To reproduce the benchmark:

```bash
python benchmarks/run.py
```

To run the coarse NEAT sweep:

```bash
python benchmarks/sweep_neat.py
```

In the reproducible Keras benchmark on the same digits family of task, tuned
NEAT reached `94.72%` mean test accuracy across three seeds, versus `97.04%`
for SGD with momentum and `96.85%` for Adam and AdamW. The detailed report is
in [`docs/research/benchmarks.md`](docs/research/benchmarks.md).

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

Useful commands:

```bash
ruff check .
ruff format .
pytest
python -m build
mkdocs build --strict
```

## Documentation

- Documentation index: [`docs/index.md`](docs/index.md)
- Quickstart: [`docs/quickstart.md`](docs/quickstart.md)
- API reference: [`docs/api.md`](docs/api.md)
- Math spec: [`docs/research/math-spec.md`](docs/research/math-spec.md)
- Player-aware mode: [`docs/research/player-aware.md`](docs/research/player-aware.md)
- Contributor guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Open Source Policy

- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Contributing guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Roadmap

- `0.1.x`: stabilize the Keras-first API and NumPy reference engine
- `0.2.x`: expand native CPU support and benchmark coverage
- `0.3.x`: add richer Keras backend coverage and adapter surface

## License

Apache-2.0. The explicit patent grant is a better default for optimization
infrastructure than MIT.
