# NEAT

NEAT, short for Nash-Equilibrium Adaptive Training, is a Keras-first optimizer
library for conflict-aware neural network optimization. It exposes a small,
production-oriented public API while keeping the novel optimizer math isolated
in a backend-agnostic engine that can be tested, benchmarked, and accelerated
independently.

## Status

- Primary public API: Keras optimizer subclass
- Reference core: NumPy
- Native acceleration: optional CPU-only C++ extension for the NumPy engine
- Tested focus for the first release line: Linux and macOS
- Python support target: 3.10+
- Keras runtime note: you must install and configure a supported Keras backend
  runtime separately, such as TensorFlow

## Installation

Core package:

```bash
pip install neat-optim
```

Keras integration:

```bash
pip install "neat-optim[keras]" tensorflow
```

Developer environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,keras]"
pytest
```

## Quick Start

Keras usage:

```python
from neat_optim import NEAT

optimizer = NEAT(
    learning_rate=3e-4,
    alpha=0.25,
    beta=0.9,
    nce_mode="projection",
)
```

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

optimizer = NEAT(learning_rate=1e-3, alpha=0.2, beta=0.9)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

Functional reference engine:

```python
import numpy as np
from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState

param = np.array([1.0, -2.0], dtype=np.float32)
grad = np.array([0.5, -0.25], dtype=np.float32)
state = ArrayState.zeros_like(param)
config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9)

result = neat_step(param, grad, state, config)
print(result.param)
print(result.metrics)
```

## Architecture

- `src/neat_optim/engine/reference.py`: canonical NumPy implementation of the
  NEAT update rule
- `src/neat_optim/engine/native.py`: optional bridge to the native CPU kernel
- `src/neat_optim/keras_optimizer.py`: Keras optimizer subclass
- `cpp/neat_core/`: pybind11-based native extension for the NumPy engine

The first release intentionally keeps the public API small:

- `NEAT`
- `NEATConfig`
- `ArrayState`
- `neat_step`

## Math Summary

For a parameter tensor `theta_t`, gradient `g_t`, momentum-like buffer
`m_{t-1}`, and correction scale `alpha`:

```text
c_t       = relu(-cos(g_t, m_{t-1}))
p_t       = proj_{m_{t-1}}(g_t)
nce_t     = -alpha * c_t * p_t
u_t       = g_t + nce_t
m_t       = beta * m_{t-1} + (1 - beta) * u_t
theta_t+1 = (1 - lr * wd) * theta_t - lr * m_t
```

NEAT uses the previous momentum buffer as the opponent proxy for the first
release. The exact specification is versioned in
[`docs/research/math-spec.md`](docs/research/math-spec.md).

## Repository Layout

```text
src/neat_optim/      Python package
cpp/neat_core/       Native CPU extension
tests/               Unit, regression, integration, and property-style tests
benchmarks/          Reproducible benchmark entrypoints
examples/            Minimal usage examples
docs/                User, research, and contributor documentation
```

## Development

Useful commands:

```bash
ruff check .
ruff format .
pytest
python -m build
```

## Roadmap

- `0.1.x`: stabilize the Keras-first API and NumPy reference engine
- `0.2.x`: ship native CPU wheels and benchmark parity
- `0.3.x`: add richer Keras backend coverage and framework adapters

## License

Apache-2.0. The explicit patent grant is a better default for optimization
infrastructure than MIT.
