# Quickstart

## Install

```bash
pip install "neat-optim[keras]" tensorflow
```

Install the core engine only:

```bash
pip install neat-optim
```

## Train With Keras

```python
import keras
from neat_optim import NEAT, NEATDiagnosticsCallback

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

model.fit(
    x_train,
    y_train,
    epochs=5,
    verbose=0,
    callbacks=[NEATDiagnosticsCallback(log_dir="runs/neat")],
)
print(optimizer.diagnostic_snapshot())
```

For Lion-style sign updates or Lookahead, opt in explicitly:

```python
optimizer = NEAT(
    learning_rate=3e-4,
    alpha=0.2,
    update_mode="lion",
    adaptive_alpha=True,
    gradient_centralization=True,
    nesterov=True,
    lookahead_k=5,
    lookahead_alpha=0.5,
)
```

## Train With Explicit Per-Example Players

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

## Use The Reference Engine

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

## Choose A Mode

- Use Keras when you want a drop-in optimizer for model training.
- Use `diagnostic_snapshot()` when you want to see whether the NEAT correction
  is actually active on your task.
- Use the reference engine when you want deterministic debugging,
  experimentation, or framework-independent validation of the update rule.
- Use the player-aware TensorFlow helper when you want each example or task to
  contribute its own gradient and you are willing to use a custom training
  loop.
