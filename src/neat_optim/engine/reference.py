from __future__ import annotations

import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.common import (
    active_fraction,
    apply_sparsity,
    as_float32,
    centralized_gradient,
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
    effective_alpha: float,
    config: NEATConfig,
) -> tuple[np.ndarray, float]:
    """Build the clipped Nash conflict correction for one gradient tensor."""
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

    correction = -effective_alpha * adaptive_scale * conflict * direction
    return clip_correction(
        correction,
        gradient,
        clip_ratio=config.nce_clip_ratio,
        eps=config.eps,
    ), conflict


def _bias_correction(beta: float, step: int) -> float:
    """Return the standard finite-step EMA correction denominator."""
    return 1.0 - (beta**step)


def _opponent_signal(
    gradient: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> np.ndarray:
    """Select the historical gradient proxy configured as the opponent."""
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


def _gradient_noise(
    gradient: np.ndarray,
    gradient_ema: np.ndarray,
    eps: float,
) -> float:
    """Measure normalized disagreement between a gradient and its EMA."""
    noise_norm = l2_norm(gradient - gradient_ema)
    scale = l2_norm(gradient) + l2_norm(gradient_ema)
    return safe_ratio(noise_norm, scale, eps)


def _effective_alpha(
    config: NEATConfig,
    conflict_ema: float,
    gradient_noise_ema: float,
    alignment_ema: float,
) -> float:
    """Adapt correction strength from conflict, noise, and alignment history."""
    if not config.adaptive_alpha:
        return float(config.alpha)
    stable_alignment = max(0.0, alignment_ema)
    scale = 1.0 + conflict_ema + gradient_noise_ema - (0.5 * stable_alignment)
    return float(
        np.clip(
            config.alpha * scale,
            config.adaptive_alpha_min,
            config.adaptive_alpha_max,
        )
    )


def neat_step_reference(
    param: np.ndarray,
    grad: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> StepResult:
    """Apply one NEAT step over NumPy arrays."""

    # Normalize all arithmetic to float32 so this executable specification is
    # deterministic across callers with different input dtypes.
    param32 = as_float32(param).copy()
    initial_param32 = param32.copy()
    grad32 = as_float32(grad)
    if config.gradient_centralization:
        grad32 = centralized_gradient(grad32)
    momentum = as_float32(state.momentum).copy()
    gradient_ema = (
        np.zeros_like(grad32)
        if state.gradient_ema is None
        else as_float32(state.gradient_ema).copy()
    )
    second_moment = (
        np.zeros_like(grad32)
        if state.second_moment is None
        else as_float32(state.second_moment).copy()
    )
    # Update the diagonal second moment before forming the preconditioner used
    # by both the correction and final parameter update.
    step_count = state.step + 1
    next_second_moment = (
        (config.second_moment_beta * second_moment)
        + ((1.0 - config.second_moment_beta) * np.square(grad32))
    ).astype(np.float32, copy=False)
    if config.bias_correction:
        second_moment_hat = next_second_moment / _bias_correction(
            config.second_moment_beta,
            step_count,
        )
    else:
        second_moment_hat = next_second_moment
    preconditioner = np.sqrt(second_moment_hat) + np.float32(config.eps)
    # Conflict is measured against historical optimization behavior rather
    # than another independently trained model.
    opponent = _opponent_signal(grad32, state, config)
    nce_gradient = grad32
    nce_opponent = opponent
    if config.adaptive_preconditioning and config.precondition_nce:
        nce_gradient = grad32 / preconditioner
        nce_opponent = opponent / preconditioner

    # These scalar EMAs drive adaptive alpha while remaining part of the saved
    # optimizer state, which makes interrupted runs reproducible.
    current_conflict = conflict_ratio(nce_gradient, nce_opponent, config.eps)
    next_conflict_ema = (
        (config.adaptive_correction_decay * state.conflict_ema)
        + ((1.0 - config.adaptive_correction_decay) * current_conflict)
    )
    current_alignment = cosine_similarity(
        grad32,
        as_float32(state.previous_gradient)
        if state.previous_gradient is not None
        else np.zeros_like(grad32),
        config.eps,
    )
    current_noise = _gradient_noise(grad32, gradient_ema, config.eps)
    next_gradient_noise_ema = (
        (config.gradient_noise_decay * state.gradient_noise_ema)
        + ((1.0 - config.gradient_noise_decay) * current_noise)
    )
    next_alignment_ema = (
        (config.gradient_noise_decay * state.alignment_ema)
        + ((1.0 - config.gradient_noise_decay) * current_alignment)
    )
    effective_alpha = _effective_alpha(
        config,
        next_conflict_ema,
        next_gradient_noise_ema,
        next_alignment_ema,
    )
    # Compute NCE in preconditioned coordinates when requested, then map it
    # back before combining it with the raw gradient.
    nce, conflict = _compute_nce(
        nce_gradient,
        nce_opponent,
        state.step,
        next_conflict_ema,
        effective_alpha,
        config,
    )
    raw_nce = (
        (nce * preconditioner).astype(np.float32, copy=False)
        if config.adaptive_preconditioning and config.precondition_nce
        else nce.astype(np.float32, copy=False)
    )
    corrected_gradient = grad32 + raw_nce
    # Momentum, Nesterov, and Lion are alternative update transports applied
    # after conflict correction; they do not change how conflict is detected.
    next_momentum = (config.beta * momentum) + (
        (1.0 - config.beta) * corrected_gradient
    )
    if config.bias_correction:
        momentum_hat = next_momentum / _bias_correction(config.beta, step_count)
    else:
        momentum_hat = next_momentum
    step_update = momentum_hat
    if config.nesterov:
        step_update = (
            (config.beta * momentum_hat) + ((1.0 - config.beta) * corrected_gradient)
        ).astype(np.float32, copy=False)
    if config.adaptive_preconditioning:
        if config.nesterov:
            step_update = step_update / preconditioner
        else:
            step_update = momentum_hat / preconditioner
        if config.nesterov:
            step_update = step_update.astype(np.float32, copy=False)
    if config.update_mode == "lion":
        step_update = np.sign(step_update).astype(np.float32, copy=False)

    # Decoupled decay scales parameters independently of the gradient update;
    # the coupled path retains classical L2-regularization semantics.
    if config.decouple_weight_decay and config.weight_decay:
        param32 *= 1.0 - (config.learning_rate * config.weight_decay)
        next_param = param32 - (config.learning_rate * step_update)
    else:
        effective_grad = step_update + (config.weight_decay * param32)
        next_param = param32 - (config.learning_rate * effective_grad)
    next_param = apply_sparsity(
        next_param,
        learning_rate=config.learning_rate,
        sparsity_l1=config.sparsity_l1,
        prune_threshold=config.prune_threshold,
    )
    # Lookahead periodically replaces fast parameters with an interpolation
    # toward a separately persisted slow copy.
    slow_param = (
        initial_param32
        if state.slow_param is None
        else as_float32(state.slow_param).copy()
    )
    if config.lookahead_k > 0 and step_count % config.lookahead_k == 0:
        slow_param = (
            slow_param
            + (config.lookahead_alpha * (next_param - slow_param))
        ).astype(np.float32, copy=False)
        next_param = slow_param.copy()

    # Construct a new state object rather than mutating the caller's state.
    # This property keeps the reference engine easy to test and reason about.
    next_state = ArrayState(
        momentum=next_momentum.astype(np.float32, copy=False),
        nce=raw_nce.astype(np.float32, copy=False),
        previous_gradient=grad32.astype(np.float32, copy=False),
        gradient_ema=update_ema(gradient_ema, grad32, config.opponent_ema_decay),
        second_moment=next_second_moment,
        slow_param=slow_param,
        conflict_ema=float(next_conflict_ema),
        gradient_noise_ema=float(next_gradient_noise_ema),
        alignment_ema=float(next_alignment_ema),
        step=state.step + 1,
    )
    grad_norm = l2_norm(grad32)
    nce_norm = l2_norm(raw_nce)
    # Diagnostics describe the actual correction and update chosen above; they
    # do not feed back into this step.
    metrics = StepMetrics(
        grad_norm=grad_norm,
        update_norm=l2_norm(step_update),
        nce_norm=nce_norm,
        conflict_ratio=conflict,
        active_fraction=active_fraction(next_param),
        opponent_norm=l2_norm(opponent),
        correction_ratio=safe_ratio(nce_norm, grad_norm, config.eps),
        update_alignment=cosine_similarity(grad32, corrected_gradient, config.eps),
        correction_active=1.0 if nce_norm > config.eps else 0.0,
        effective_alpha=effective_alpha,
        gradient_noise=current_noise,
    )
    return StepResult(
        param=next_param.astype(np.float32, copy=False),
        state=next_state,
        metrics=metrics,
    )
