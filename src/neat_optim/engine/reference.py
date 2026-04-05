from __future__ import annotations

import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.common import (
    active_fraction,
    apply_sparsity,
    as_float32,
    clip_correction,
    conflict_ratio,
    cosine_similarity,
    safe_projection,
    safe_ratio,
    update_ema,
)
from neat_optim.state import ArrayState, StepMetrics, StepResult
from neat_optim.utils.metrics import l2_norm


def _safe_projection(
    gradient: np.ndarray, vector: np.ndarray, eps: float
) -> np.ndarray:
    return safe_projection(gradient, vector, eps)


def _conflict_ratio(gradient: np.ndarray, vector: np.ndarray, eps: float) -> float:
    return conflict_ratio(gradient, vector, eps)


def _compute_nce(
    gradient: np.ndarray,
    opponent: np.ndarray,
    state_step: int,
    conflict_ema: float,
    config: NEATConfig,
) -> tuple[np.ndarray, float]:
    if config.nce_mode == "off":
        return np.zeros_like(gradient), 0.0
    if state_step < config.correction_warmup_steps:
        return np.zeros_like(gradient), 0.0

    conflict = _conflict_ratio(gradient, opponent, config.eps)
    if conflict < config.conflict_threshold:
        return np.zeros_like(gradient), conflict
    if config.nce_mode == "cosine":
        direction = gradient
    else:
        direction = _safe_projection(gradient, opponent, config.eps)

    adaptive_scale = np.float32(1.0)
    if config.adaptive_correction:
        grad_norm = l2_norm(gradient)
        opponent_norm = l2_norm(opponent)
        reliability = opponent_norm / (grad_norm + opponent_norm + config.eps)
        adaptive_scale = np.float32(
            np.clip(
                1.0 + reliability + max(conflict, conflict_ema),
                config.adaptive_correction_min_scale,
                config.adaptive_correction_max_scale,
            )
        )

    correction = -config.alpha * adaptive_scale * conflict * direction
    return clip_correction(
        correction,
        gradient,
        clip_ratio=config.nce_clip_ratio,
        eps=config.eps,
    ), conflict


def _opponent_signal(
    gradient: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> np.ndarray:
    if config.opponent_source == "blended":
        momentum = as_float32(state.momentum)
        ema = (
            np.zeros_like(gradient)
            if state.gradient_ema is None
            else as_float32(state.gradient_ema)
        )
        blend = np.float32(config.opponent_blend)
        return (blend * momentum) + ((np.float32(1.0) - blend) * ema)
    if config.opponent_source == "previous_gradient":
        previous = state.previous_gradient
        return np.zeros_like(gradient) if previous is None else as_float32(previous)
    if config.opponent_source == "gradient_ema":
        ema = state.gradient_ema
        return np.zeros_like(gradient) if ema is None else as_float32(ema)
    return as_float32(state.momentum)


def neat_step_reference(
    param: np.ndarray,
    grad: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> StepResult:
    """Apply one NEAT step over NumPy arrays."""

    param32 = as_float32(param).copy()
    grad32 = as_float32(grad)
    momentum = as_float32(state.momentum).copy()
    gradient_ema = (
        np.zeros_like(grad32)
        if state.gradient_ema is None
        else as_float32(state.gradient_ema).copy()
    )
    opponent = _opponent_signal(grad32, state, config)
    current_conflict = conflict_ratio(grad32, opponent, config.eps)
    next_conflict_ema = (
        (config.adaptive_correction_decay * state.conflict_ema)
        + ((1.0 - config.adaptive_correction_decay) * current_conflict)
    )
    nce, conflict = _compute_nce(
        grad32,
        opponent,
        state.step,
        next_conflict_ema,
        config,
    )
    update_direction = grad32 + nce
    next_momentum = (config.beta * momentum) + ((1.0 - config.beta) * update_direction)

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
        nce=nce.astype(np.float32, copy=False),
        previous_gradient=grad32.astype(np.float32, copy=False),
        gradient_ema=update_ema(gradient_ema, grad32, config.opponent_ema_decay),
        conflict_ema=float(next_conflict_ema),
        step=state.step + 1,
    )
    grad_norm = l2_norm(grad32)
    nce_norm = l2_norm(nce)
    metrics = StepMetrics(
        grad_norm=grad_norm,
        update_norm=l2_norm(next_momentum),
        nce_norm=nce_norm,
        conflict_ratio=conflict,
        active_fraction=active_fraction(next_param),
        opponent_norm=l2_norm(opponent),
        correction_ratio=safe_ratio(nce_norm, grad_norm, config.eps),
        update_alignment=cosine_similarity(grad32, update_direction, config.eps),
        correction_active=1.0 if nce_norm > config.eps else 0.0,
    )
    return StepResult(
        param=next_param.astype(np.float32, copy=False),
        state=next_state,
        metrics=metrics,
    )
