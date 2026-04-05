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
- `sparsity_l1`: optional soft-threshold shrinkage for sparse weights
- `prune_threshold`: optional hard pruning threshold

Diagnostics:

- `diagnostic_snapshot()`: returns aggregate optimizer diagnostics gathered
  during training
- `reset_diagnostics()`: clears those running aggregates

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
