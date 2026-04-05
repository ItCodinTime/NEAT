from __future__ import annotations

import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.state import ArrayState, StepMetrics, StepResult
from neat_optim.utils.metrics import l2_norm


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _flat_dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left.reshape(-1), right.reshape(-1)))


def _safe_projection(
    gradient: np.ndarray, vector: np.ndarray, eps: float
) -> np.ndarray:
    denom = _flat_dot(vector, vector)
    if denom <= eps:
        return np.zeros_like(gradient)
    return (_flat_dot(gradient, vector) / (denom + eps)) * vector


def _conflict_ratio(gradient: np.ndarray, vector: np.ndarray, eps: float) -> float:
    grad_norm = l2_norm(gradient)
    vec_norm = l2_norm(vector)
    if grad_norm <= eps or vec_norm <= eps:
        return 0.0
    cosine = _flat_dot(gradient, vector) / ((grad_norm * vec_norm) + eps)
    return max(0.0, -cosine)


def _compute_nce(
    gradient: np.ndarray, momentum: np.ndarray, config: NEATConfig
) -> tuple[np.ndarray, float]:
    if config.nce_mode == "off":
        return np.zeros_like(gradient), 0.0
    if config.nce_mode == "cosine":
        conflict_ratio = _conflict_ratio(gradient, momentum, config.eps)
        direction = gradient
    else:
        conflict_ratio = _conflict_ratio(gradient, momentum, config.eps)
        direction = _safe_projection(gradient, momentum, config.eps)

    correction = -config.alpha * conflict_ratio * direction
    grad_norm = l2_norm(gradient)
    correction_norm = l2_norm(correction)
    clip_limit = config.nce_clip_ratio * grad_norm
    if correction_norm > clip_limit and correction_norm > config.eps:
        correction = correction * (clip_limit / correction_norm)
    return correction.astype(np.float32, copy=False), conflict_ratio


def neat_step_reference(
    param: np.ndarray,
    grad: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> StepResult:
    """Apply one NEAT step over NumPy arrays."""

    param32 = _as_float32(param).copy()
    grad32 = _as_float32(grad)
    momentum = _as_float32(state.momentum).copy()

    nce, conflict_ratio = _compute_nce(grad32, momentum, config)
    update_direction = grad32 + nce
    next_momentum = (config.beta * momentum) + ((1.0 - config.beta) * update_direction)

    if config.decouple_weight_decay and config.weight_decay:
        param32 *= 1.0 - (config.learning_rate * config.weight_decay)
        next_param = param32 - (config.learning_rate * next_momentum)
    else:
        effective_grad = next_momentum + (config.weight_decay * param32)
        next_param = param32 - (config.learning_rate * effective_grad)

    next_state = ArrayState(
        momentum=next_momentum.astype(np.float32, copy=False),
        nce=nce.astype(np.float32, copy=False),
        step=state.step + 1,
    )
    metrics = StepMetrics(
        grad_norm=l2_norm(grad32),
        update_norm=l2_norm(next_momentum),
        nce_norm=l2_norm(nce),
        conflict_ratio=conflict_ratio,
    )
    return StepResult(
        param=next_param.astype(np.float32, copy=False),
        state=next_state,
        metrics=metrics,
    )
