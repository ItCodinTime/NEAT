"""Shared numerical helpers for NEAT engines."""

from __future__ import annotations

import numpy as np

from neat_optim.utils.metrics import l2_norm


def as_float32(array: np.ndarray) -> np.ndarray:
    """Return `array` as a float32 NumPy array."""
    return np.asarray(array, dtype=np.float32)


def flat_dot(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a dot product over arbitrarily shaped tensors."""
    return float(np.dot(left.reshape(-1), right.reshape(-1)))


def safe_ratio(numerator: float, denominator: float, eps: float) -> float:
    """Return a numerically safe ratio."""
    return float(numerator / (denominator + eps))


def safe_projection(
    gradient: np.ndarray,
    vector: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Project `gradient` onto `vector`, returning zeros for near-zero vectors."""
    denom = flat_dot(vector, vector)
    if denom <= eps:
        return np.zeros_like(gradient)
    return (flat_dot(gradient, vector) / (denom + eps)) * vector


def conflict_ratio(gradient: np.ndarray, vector: np.ndarray, eps: float) -> float:
    """Return the negative cosine conflict ratio between two tensors."""
    grad_norm = l2_norm(gradient)
    vec_norm = l2_norm(vector)
    if grad_norm <= eps or vec_norm <= eps:
        return 0.0
    cosine = flat_dot(gradient, vector) / ((grad_norm * vec_norm) + eps)
    return max(0.0, -cosine)


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float) -> float:
    """Return cosine similarity between arbitrarily shaped tensors."""
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm <= eps or right_norm <= eps:
        return 0.0
    return flat_dot(left, right) / ((left_norm * right_norm) + eps)


def clip_correction(
    correction: np.ndarray,
    gradient: np.ndarray,
    clip_ratio: float,
    eps: float,
) -> np.ndarray:
    """Clip a correction term relative to the base gradient norm."""
    grad_norm = l2_norm(gradient)
    correction_norm = l2_norm(correction)
    clip_limit = clip_ratio * grad_norm
    if correction_norm > clip_limit and correction_norm > eps:
        correction = correction * (clip_limit / correction_norm)
    return correction.astype(np.float32, copy=False)


def apply_sparsity(
    param: np.ndarray,
    learning_rate: float,
    sparsity_l1: float,
    prune_threshold: float,
) -> np.ndarray:
    """Apply soft-threshold sparsity and optional hard pruning."""
    next_param = param.astype(np.float32, copy=False)
    if sparsity_l1 > 0.0:
        shrink = np.float32(learning_rate * sparsity_l1)
        next_param = np.sign(next_param) * np.maximum(np.abs(next_param) - shrink, 0.0)
    if prune_threshold > 0.0:
        next_param = np.where(np.abs(next_param) < prune_threshold, 0.0, next_param)
    return next_param.astype(np.float32, copy=False)


def active_fraction(array: np.ndarray) -> float:
    """Return the fraction of non-zero elements in `array`."""
    if array.size == 0:
        return 1.0
    return float(np.count_nonzero(array) / array.size)


def update_ema(previous: np.ndarray, current: np.ndarray, decay: float) -> np.ndarray:
    """Return an EMA update in float32."""
    return (
        (np.float32(decay) * previous) + (np.float32(1.0 - decay) * current)
    ).astype(np.float32, copy=False)


def centralized_gradient(gradient: np.ndarray) -> np.ndarray:
    """Apply gradient centralization to matrix-like tensors."""
    gradient32 = as_float32(gradient)
    if gradient32.ndim <= 1:
        return gradient32
    axes = tuple(range(gradient32.ndim - 1))
    return (gradient32 - np.mean(gradient32, axis=axes, keepdims=True)).astype(
        np.float32,
        copy=False,
    )
