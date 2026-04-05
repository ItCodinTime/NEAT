"""TensorFlow helpers for explicit player-aware NEAT training."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from neat_optim.config import PlayerNEATConfig
from neat_optim.engine.multiplayer import neat_player_step
from neat_optim.state import ArrayState, PlayerStepMetrics


def _tensorflow():
    try:
        return import_module("tensorflow")
    except ImportError as exc:  # pragma: no cover - import guard behavior
        raise ImportError(
            "Player-aware TensorFlow helpers require TensorFlow to be installed."
        ) from exc


@dataclass(slots=True)
class TensorFlowPlayerStepResult:
    """Return value for a single TensorFlow player-aware training step."""

    loss: float
    states: list[ArrayState]
    metrics: list[PlayerStepMetrics]


def create_player_states(model: Any) -> list[ArrayState]:
    """Create `ArrayState` entries matching a built TensorFlow/Keras model."""
    variables = list(model.trainable_variables)
    if not variables:
        raise ValueError("model must be built before creating player states.")
    return [ArrayState.zeros_like(variable.numpy()) for variable in variables]


def player_train_step(
    model: Any,
    x: Any,
    y: Any,
    loss_fn: Any,
    states: list[ArrayState] | None,
    config: PlayerNEATConfig,
) -> TensorFlowPlayerStepResult:
    """Run one explicit per-example player-aware training step.

    `loss_fn` must return unreduced per-example losses. For Keras losses, use
    `reduction="none"`.
    """

    tf = _tensorflow()
    variables = list(model.trainable_variables)
    if not variables:
        raise ValueError("model must be built before calling player_train_step.")
    if states is None:
        states = create_player_states(model)
    if len(states) != len(variables):
        raise ValueError("states must match model.trainable_variables in length.")

    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x, training=True)
        per_player_loss = tf.convert_to_tensor(loss_fn(y, predictions))
        if per_player_loss.shape.rank == 0:
            raise ValueError(
                "loss_fn must return per-example losses; use reduction='none'."
            )
        per_player_loss = tf.reshape(
            per_player_loss,
            (tf.shape(per_player_loss)[0], -1),
        )
        per_player_loss = tf.reduce_mean(per_player_loss, axis=1)
        batch_loss = tf.reduce_mean(per_player_loss)

    next_states: list[ArrayState] = []
    metrics: list[PlayerStepMetrics] = []
    for variable, state in zip(variables, states, strict=True):
        player_grads = tape.jacobian(
            per_player_loss,
            variable,
            experimental_use_pfor=False,
        )
        if player_grads is None:
            next_states.append(state)
            continue
        result = neat_player_step(
            variable.numpy(),
            tf.cast(player_grads, tf.float32).numpy(),
            state,
            config,
        )
        variable.assign(tf.convert_to_tensor(result.param, dtype=variable.dtype))
        next_states.append(result.state)
        metrics.append(result.metrics)

    del tape
    return TensorFlowPlayerStepResult(
        loss=float(batch_loss.numpy()),
        states=next_states,
        metrics=metrics,
    )
