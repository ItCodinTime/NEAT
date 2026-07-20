"""Task-aware NEAT gradient transformation for PyTorch multi-task models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Task-aware NEAT requires PyTorch. Install `neat-optim[torch]`."
    ) from exc


@dataclass(frozen=True, slots=True)
class TaskAwareNEATResult:
    """Combined gradients and device-side diagnostics from one transformation."""

    gradients: tuple[Any, ...]
    task_conflicts: tuple[Any, ...]
    correction_ratio: Any
    gradient_alignment: Any
    active_task_fraction: Any
    minimum_task_alignment: Any


def _dot(left, right, selected: tuple[bool, ...]):
    return sum(
        (first.float() * second.float()).sum()
        for first, second, include in zip(left, right, selected, strict=True)
        if include
    )


def _mean(rows: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    return tuple(sum(values) / len(rows) for values in zip(*rows, strict=True))


def _validate_rows(task_gradients, shared_parameters, alpha: float, eps: float):
    if len(task_gradients) < 2:
        raise ValueError("task-aware NEAT requires at least two task gradients")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    width = len(task_gradients[0])
    if width == 0 or any(len(row) != width for row in task_gradients):
        raise ValueError("all task-gradient rows must have the same non-zero length")
    for tensors in zip(*task_gradients, strict=True):
        reference = tensors[0]
        if any(tensor.shape != reference.shape for tensor in tensors[1:]):
            raise ValueError("corresponding task gradients must have matching shapes")
    selected = (
        tuple(True for _ in range(width))
        if shared_parameters is None
        else tuple(bool(value) for value in shared_parameters)
    )
    if len(selected) != width:
        raise ValueError("shared_parameters must match the task-gradient row length")
    if not any(selected):
        raise ValueError("at least one parameter tensor must be marked shared")
    return selected


def task_aware_neat(
    task_gradients,
    *,
    alpha: float = 0.25,
    shared_parameters=None,
    eps: float = 1e-12,
) -> TaskAwareNEATResult:
    """Combine per-task gradients with a global, bounded soft projection.

    Each task is compared with the mean of the other tasks over shared parameter
    tensors. When the two directions conflict, ``alpha`` interpolates between the
    raw task gradient and its full conflict projection. Task-specific tensors are
    averaged unchanged. The returned direction is intended to be passed to a
    conventional optimizer such as AdamW or Lion.
    """

    rows = tuple(tuple(row) for row in task_gradients)
    selected = _validate_rows(rows, shared_parameters, alpha, eps)
    raw = _mean(rows)
    corrected_rows = []
    conflicts = []
    for index, gradient in enumerate(rows):
        peers = tuple(row for peer, row in enumerate(rows) if peer != index)
        opponent = _mean(peers)
        inner = _dot(gradient, opponent, selected)
        gradient_norm = _dot(gradient, gradient, selected).sqrt()
        opponent_norm_squared = _dot(opponent, opponent, selected)
        opponent_norm = opponent_norm_squared.sqrt()
        conflict = torch.clamp(
            -inner / (gradient_norm * opponent_norm + eps), min=0.0, max=1.0
        )
        # Projection magnitude already scales linearly with conflict angle. Do
        # not multiply by the cosine again, which would attenuate it quadratically.
        projection_scale = torch.minimum(inner, torch.zeros_like(inner)) / (
            opponent_norm_squared + eps
        )
        corrected_rows.append(
            tuple(
                value - alpha * projection_scale * peer_value if include else value
                for value, peer_value, include in zip(
                    gradient, opponent, selected, strict=True
                )
            )
        )
        conflicts.append(conflict)

    corrected = _mean(tuple(corrected_rows))
    delta = tuple(after - before for before, after in zip(raw, corrected, strict=True))
    raw_norm = _dot(raw, raw, selected).sqrt()
    corrected_norm = _dot(corrected, corrected, selected).sqrt()
    correction_ratio = _dot(delta, delta, selected).sqrt() / (raw_norm + eps)
    gradient_alignment = _dot(raw, corrected, selected) / (
        raw_norm * corrected_norm + eps
    )
    task_alignments = tuple(
        _dot(row, corrected, selected)
        / (_dot(row, row, selected).sqrt() * corrected_norm + eps)
        for row in rows
    )
    conflict_stack = torch.stack(conflicts)
    return TaskAwareNEATResult(
        gradients=corrected,
        task_conflicts=tuple(conflicts),
        correction_ratio=correction_ratio,
        gradient_alignment=gradient_alignment,
        active_task_fraction=(conflict_stack > 0.0).float().mean(),
        minimum_task_alignment=torch.stack(task_alignments).min(),
    )
