# Quickstart

## Installation

### With TensorFlow backend

```bash
pip install "neat-optim[keras]" tensorflow
```

### With PyTorch backend

```bash
pip install "neat-optim[keras]" torch
```

### With JAX backend

```bash
pip install "neat-optim[keras]" jax
```

### NumPy core only (no Keras required)

```bash
pip install neat-optim
```

The `neat-optim` core depends only on NumPy. Keras and a backend runtime are optional and only needed for the `NEAT` Keras optimizer.

---

## Train a Keras Model

The fastest way to use NEAT is as a drop-in optimizer in `model.compile`:

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

### Check whether the correction is engaging

After training (or after each epoch), call `diagnostic_snapshot()` to see how active the NEAT correction was:

```python
print(model.optimizer.diagnostic_snapshot())
# {
#   "mean_conflict_ratio": 0.021,
#   "mean_correction_ratio": 0.008,
#   "mean_update_alignment": 0.997,
#   "mean_opponent_norm": 0.043,
#   "correction_active_fraction": 0.74,
# }
```

If `mean_conflict_ratio` and `correction_active_fraction` are both near zero, the correction is not engaging on this task. That is normal for tasks with well-aligned gradients — standard Adam may be equally effective.

Reset the accumulators at the start of each epoch to get per-epoch diagnostics:

```python
for epoch in range(epochs):
    model.optimizer.reset_diagnostics()
    model.fit(x_train, y_train, epochs=1)
    print(f"Epoch {epoch}:", model.optimizer.diagnostic_snapshot())
```

---

## Configure the Optimizer

All parameters have sensible defaults. The most useful ones to tune:

```python
optimizer = NEAT(
    learning_rate=1e-3,

    # Correction strength — higher values remove more of the conflicting component.
    alpha=0.25,

    # Momentum decay — same role as in Adam.
    beta=0.9,

    # NCE mode: "projection" (default), "cosine", or "off".
    nce_mode="projection",

    # What signal to use as the conflict opponent.
    # "momentum" (default), "gradient_ema", "previous_gradient", or "blended".
    opponent_source="gradient_ema",

    # Wait this many steps before the correction activates.
    correction_warmup_steps=5,

    # Only correct when conflict ratio exceeds this threshold.
    conflict_threshold=0.05,

    # Adam-style second-moment preconditioning.
    adaptive_preconditioning=False,

    # Soft L1 sparsity pressure after each step.
    sparsity_l1=0.0,
)
```

See the [API reference](api.md) for the full parameter list.

---

## Use the NumPy Reference Engine

The reference engine works without Keras and is useful for debugging, experimentation, or framework-independent testing:

```python
import numpy as np
from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState

param  = np.array([1.0, -2.0], dtype=np.float32)
grad   = np.array([0.5, -0.25], dtype=np.float32)
state  = ArrayState.zeros_like(param)
config = NEATConfig(learning_rate=0.1, alpha=0.25, beta=0.9, nce_mode="projection")

result = neat_step(param, grad, state, config)
print(result.param)           # updated parameters
print(result.state.momentum)  # updated momentum
print(result.metrics)         # conflict_ratio, nce_norm, update_alignment, ...

# Carry state forward for the next step
state = result.state
```

---

## Per-Example Player-Aware Training

Player-aware mode treats each example's gradient as a separate player and builds the conflict opponent from the other examples in the batch. It requires a custom training loop:

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

# Initialize one optimizer state per model variable, per player
states = create_player_states(model)

config = PlayerNEATConfig(
    learning_rate=1e-2,
    alpha=0.25,
    beta=0.9,
    sparsity_l1=1e-4,     # soft sparsity pressure
    prune_threshold=1e-3,  # hard prune small weights
)

for x_batch, y_batch in dataset:
    result = player_train_step(model, x_batch, y_batch, loss_fn, states, config)
    states = result.states   # carry forward
    print(f"loss: {result.loss:.4f}")
```

Player-aware mode uses `tf.GradientTape(persistent=True)` and `tape.jacobian()` internally, so it requires TensorFlow as the Keras backend. See [`docs/research/player-aware.md`](research/player-aware.md) for details on how the opponent proxy is constructed.

---

## Which Mode Should I Use?

| Situation | Recommended mode |
|---|---|
| Standard training with `model.fit` | Keras optimizer (`NEAT`) |
| Custom training loop, aggregated gradients | Keras optimizer with `apply_gradients` |
| Per-example gradient conflict modeling | Player-aware mode |
| Framework-agnostic debugging or testing | NumPy reference engine |
| Checking if correction is active on your task | Any mode + `diagnostic_snapshot()` |
