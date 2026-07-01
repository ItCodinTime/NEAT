# API

## Public Symbols

### `neat_optim.NEAT`

Keras optimizer implementation of NEAT.

Use this when training a Keras model through `model.compile(...)`.

Important constructor arguments:

- `learning_rate`: step size
- `alpha`: scale of the Nash correction estimate
- `beta`: momentum mixing factor
- `nce_mode`: one of `projection`, `cosine`, or `off`
- `weight_decay`: weight decay coefficient
- `decouple_weight_decay`: whether weight decay is decoupled from the momentum
  update
- `nce_clip_ratio`: clips the correction norm relative to the gradient norm
- `opponent_source`: `momentum`, `previous_gradient`, or `gradient_ema`
- `opponent_ema_decay`: decay factor used by `gradient_ema`
- `correction_warmup_steps`: steps to wait before applying the correction
- `conflict_threshold`: minimum conflict ratio required before correction
- `adaptive_correction`: scales the correction using conflict and opponent
  reliability
- `adaptive_preconditioning`: enables the diagonal second-moment
  preconditioner
- `precondition_nce`: computes the correction in the preconditioned space
- `update_mode`: `momentum` or `lion`; `lion` applies a sign-based final
  update
- `adaptive_alpha`: makes `alpha` dynamic from conflict, gradient noise, and
  alignment EMAs
- `adaptive_alpha_min` / `adaptive_alpha_max`: clamps the dynamic alpha
- `gradient_noise_decay`: EMA decay for gradient-noise and alignment signals
- `gradient_centralization`: subtracts the feature mean from matrix-like
  gradients before the optimizer update
- `nesterov`: uses a Nesterov-style lookahead momentum update
- `lookahead_k` / `lookahead_alpha`: optional Lookahead slow-weight sync
- `sparsity_l1`: optional soft-threshold shrinkage for sparse weights
- `prune_threshold`: optional hard pruning threshold

Diagnostics:

- `diagnostic_snapshot()`: returns aggregate optimizer diagnostics gathered
  during training
- `reset_diagnostics()`: clears those running aggregates

Snapshot keys:

- `mean_conflict_ratio`
- `mean_correction_ratio`
- `mean_update_alignment`
- `mean_opponent_norm`
- `correction_active_fraction`
- `mean_effective_alpha`
- `mean_gradient_noise`

### `neat_optim.TorchNEAT`

Optional PyTorch optimizer adapter used by the modern benchmark suite. Install
the `torch` extra before importing it.

```python
from neat_optim import TorchNEAT

optimizer = TorchNEAT(
    model.parameters(),
    lr=1e-3,
    opponent_source="previous_gradient",
    adaptive_alpha=True,
    adaptive_preconditioning=True,
    diagnostic_interval=10,
)
```

`diagnostic_interval` controls how often parameter-level diagnostic reductions
are sampled. It defaults to `1` for complete measurement. Larger values reduce
accelerator synchronization overhead without changing parameter updates. The
adapter supports normal PyTorch parameter groups, `state_dict()` checkpointing,
closures, and the same algorithm settings validated by `NEATConfig`.

### `neat_optim.NEATDiagnosticsCallback`

Keras callback for optimizer diagnostics.

Use this when you want epoch-level diagnostics in memory or TensorBoard:

```python
from neat_optim import NEATDiagnosticsCallback

callbacks = [
    NEATDiagnosticsCallback(log_dir="runs/neat", reset_each_epoch=True),
]
model.fit(x_train, y_train, callbacks=callbacks)
```

If `log_dir` is set, TensorFlow must be installed because the callback writes
with `tf.summary`. Without `log_dir`, the callback stores snapshots in
`callback.history`.

### `neat_optim.NEATConfig`

Immutable configuration object for the reference and functional engines.

Use this when working with `neat_step(...)` directly or when serializing
optimizer configuration outside Keras.

### `neat_optim.ArrayState`

Per-parameter state container for the NumPy and functional engines.

Fields:

- `momentum`
- `nce`
- `previous_gradient`
- `gradient_ema`
- `second_moment`
- `slow_param`
- `conflict_ema`
- `gradient_noise_ema`
- `alignment_ema`
- `step`

Create a zero-initialized state with `ArrayState.zeros_like(array)`.

### `neat_optim.PlayerNEATConfig`

Configuration object for explicit per-player updates.

Additional arguments:

- `opponent_mode`: `mean_excluding_self` or `batch_mean`
- `player_reduction`: `mean` or `sum`
- `sparsity_l1`: soft-threshold sparsity pressure
- `prune_threshold`: hard-prune threshold

### `neat_optim.engine.multiplayer.neat_player_step`

Explicit player-aware one-step API.

Inputs:

- parameter array
- `player_grads` with shape `(num_players, *param.shape)`
- `ArrayState`
- `PlayerNEATConfig`

Returns a `PlayerStepResult` containing:

- updated parameter tensor
- updated state
- player-conflict and sparsity diagnostics

### `neat_optim.training.player_train_step`

TensorFlow helper for custom training loops with per-example gradients.

Use this when you want each training example in a batch to act as a player.
This requires a loss function that returns unreduced per-example losses.

### `neat_optim.engine.functional.neat_step`

Framework-agnostic one-step API for NEAT updates.

Inputs:

- parameter array
- gradient array
- `ArrayState`
- `NEATConfig`

Returns a `StepResult` containing:

- updated parameter tensor
- updated state
- scalar metrics for the step, including conflict ratio, correction ratio,
  update alignment, and opponent norm

## Serialization

- `NEAT.get_config()` supports standard Keras optimizer serialization.
- `NEATConfig.as_dict()` returns a plain serializable configuration mapping.
- `PlayerNEATConfig.as_dict()` does the same for the player-aware mode.
- `ArrayState` is intended for the reference NumPy engine and test fixtures,
  not for framework checkpoint interchange.

## Exceptions

- `ConfigurationError`: invalid optimizer configuration
- `NativeCoreUnavailableError`: native engine requested but not available
