"""Player-aware NEAT stepping over per-player gradient tensors."""

from __future__ import annotations

import numpy as np

from neat_optim.config import PlayerNEATConfig
from neat_optim.engine.common import (
    active_fraction,
    apply_sparsity,
    as_float32,
    clip_correction,
    conflict_ratio,
    safe_projection,
    safe_ratio,
)
from neat_optim.exceptions import ShapeError
from neat_optim.state import ArrayState, PlayerStepMetrics, PlayerStepResult
from neat_optim.utils.metrics import l2_norm


def _validate_player_grads(param: np.ndarray, player_grads: np.ndarray) -> None:
    """Enforce the leading-player-axis contract before numerical work."""
    if player_grads.ndim < 1:
        raise ShapeError("player_grads must have a leading player dimension.")
    if tuple(player_grads.shape[1:]) != tuple(param.shape):
        raise ShapeError(
            "player_grads must have shape (num_players, *param.shape)."
        )


def _player_opponent(
    player_grads: np.ndarray,
    index: int,
    config: PlayerNEATConfig,
    summed_grads: np.ndarray,
) -> np.ndarray:
    """Return the peer-gradient signal faced by one player."""
    if config.opponent_mode == "batch_mean":
        return player_grads.mean(axis=0)
    if player_grads.shape[0] == 1:
        return np.zeros_like(player_grads[index])
    return (summed_grads - player_grads[index]) / np.float32(player_grads.shape[0] - 1)


def _aggregate_players(
    tensors: np.ndarray,
    reduction: str,
) -> np.ndarray:
    """Combine player tensors using the configured batch reduction."""
    if reduction == "sum":
        return tensors.sum(axis=0)
    return tensors.mean(axis=0)


def neat_player_step(
    param: np.ndarray,
    player_grads: np.ndarray,
    state: ArrayState,
    config: PlayerNEATConfig,
) -> PlayerStepResult:
    """Apply one explicit player-aware NEAT step.

    `player_grads` is expected to contain one gradient tensor per player or
    training example. Each player's opponent signal is constructed from the
    remaining players, a correction is applied, and the corrected gradients are
    then aggregated into the batch update.
    """

    param32 = as_float32(param).copy()
    player_grads32 = as_float32(player_grads)
    _validate_player_grads(param32, player_grads32)
    momentum = as_float32(state.momentum).copy()
    next_conflict_ema = state.conflict_ema

    # Cache the batch sum so mean-excluding-self opponents remain O(players)
    # rather than repeatedly summing all peers inside the loop.
    summed_grads = player_grads32.sum(axis=0)
    conflicts = []
    corrections = []
    corrected_players = []
    correction_ratios = []
    # Correct every player independently. Aggregating first would erase the
    # pairwise conflicts that this mode is specifically intended to retain.
    for index, gradient in enumerate(player_grads32):
        opponent = _player_opponent(player_grads32, index, config, summed_grads)
        conflict = conflict_ratio(gradient, opponent, config.eps)
        next_conflict_ema = (
            (config.adaptive_correction_decay * next_conflict_ema)
            + ((1.0 - config.adaptive_correction_decay) * conflict)
        )
        if (
            config.nce_mode == "off"
            or state.step < config.correction_warmup_steps
            or conflict < config.conflict_threshold
        ):
            correction = np.zeros_like(gradient)
        else:
            direction = (
                gradient
                if config.nce_mode == "cosine"
                else safe_projection(gradient, opponent, config.eps)
            )
            adaptive_scale = 1.0
            if config.adaptive_correction:
                grad_norm = l2_norm(gradient)
                opponent_norm = l2_norm(opponent)
                reliability = opponent_norm / (grad_norm + opponent_norm + config.eps)
                adaptive_scale = float(
                    np.clip(
                        1.0 + reliability + max(conflict, next_conflict_ema),
                        config.adaptive_correction_min_scale,
                        config.adaptive_correction_max_scale,
                    )
                )
            correction = -config.alpha * adaptive_scale * conflict * direction
            correction = clip_correction(
                correction,
                gradient,
                clip_ratio=config.nce_clip_ratio,
                eps=config.eps,
            )
        conflicts.append(conflict)
        corrections.append(correction)
        correction_ratios.append(
            safe_ratio(l2_norm(correction), l2_norm(gradient), config.eps)
        )
        corrected_players.append((gradient + correction).astype(np.float32, copy=False))

    # Reduction happens only after each player's correction is finalized.
    correction_stack = np.stack(corrections, axis=0)
    corrected_stack = np.stack(corrected_players, axis=0)
    aggregate_grad = _aggregate_players(player_grads32, config.player_reduction)
    aggregate_correction = _aggregate_players(correction_stack, config.player_reduction)
    aggregate_update = _aggregate_players(corrected_stack, config.player_reduction)
    next_momentum = (config.beta * momentum) + ((1.0 - config.beta) * aggregate_update)

    # Regularization is parameter-level, so it is applied once after player
    # gradients have been reduced.
    if config.decouple_weight_decay and config.weight_decay:
        param32 *= 1.0 - (config.learning_rate * config.weight_decay)
        next_param = param32 - (config.learning_rate * next_momentum)
    else:
        effective_grad = next_momentum + (config.weight_decay * param32)
        next_param = param32 - (config.learning_rate * effective_grad)
    next_param = apply_sparsity(
        next_param,
        learning_rate=config.learning_rate,
        sparsity_l1=config.sparsity_l1,
        prune_threshold=config.prune_threshold,
    )

    next_state = ArrayState(
        momentum=next_momentum.astype(np.float32, copy=False),
        nce=aggregate_correction.astype(np.float32, copy=False),
        conflict_ema=float(next_conflict_ema),
        step=state.step + 1,
    )
    metrics = PlayerStepMetrics(
        grad_norm=l2_norm(aggregate_grad),
        update_norm=l2_norm(next_momentum),
        nce_norm=l2_norm(aggregate_correction),
        mean_player_conflict=float(np.mean(conflicts)),
        max_player_conflict=float(np.max(conflicts)),
        active_fraction=active_fraction(next_param),
        num_players=int(player_grads32.shape[0]),
        mean_correction_ratio=float(np.mean(correction_ratios)),
    )
    return PlayerStepResult(
        param=next_param.astype(np.float32, copy=False),
        state=next_state,
        metrics=metrics,
    )
