"""Torch-native two-task gradient combination rules used by STS benchmarks."""

from __future__ import annotations

from collections.abc import Sequence


def _dot(left: Sequence, right: Sequence):
    return sum((a * b).sum() for a, b in zip(left, right, strict=True))


def _zeros_for_unused(gradients: Sequence, parameters: Sequence):
    import torch

    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(gradients, parameters, strict=True)
    )


def task_gradients(losses: Sequence, parameters: Sequence) -> tuple[tuple, ...]:
    """Return one dense gradient tuple per task without modifying ``.grad``."""
    import torch

    rows = []
    for index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(losses) - 1,
            allow_unused=True,
        )
        rows.append(_zeros_for_unused(gradients, parameters))
    return tuple(rows)


def mean_gradients(task_rows: Sequence[Sequence]) -> tuple:
    """Return the arithmetic mean task gradient for every parameter."""
    return tuple(
        sum(values) / len(task_rows) for values in zip(*task_rows, strict=True)
    )


def pcgrad_two_task(task_rows: Sequence[Sequence], *, eps: float = 1e-12) -> tuple:
    """Apply the two-task specialization of PCGrad and average the projections."""
    if len(task_rows) != 2:
        raise ValueError("PCGrad benchmark implementation requires exactly two tasks")
    first, second = task_rows
    inner = _dot(first, second)
    first_norm = _dot(first, first)
    second_norm = _dot(second, second)
    if float(inner.detach()) < 0.0:
        projected_first = tuple(
            left - inner / (second_norm + eps) * right
            for left, right in zip(first, second, strict=True)
        )
        projected_second = tuple(
            right - inner / (first_norm + eps) * left
            for left, right in zip(first, second, strict=True)
        )
    else:
        projected_first, projected_second = first, second
    return tuple(
        (left + right) / 2.0
        for left, right in zip(projected_first, projected_second, strict=True)
    )


def mgda_two_task(task_rows: Sequence[Sequence], *, eps: float = 1e-12) -> tuple:
    """Return the exact minimum-norm convex combination of two task gradients."""
    import torch

    if len(task_rows) != 2:
        raise ValueError("MGDA benchmark implementation requires exactly two tasks")
    first, second = task_rows
    difference = tuple(left - right for left, right in zip(first, second, strict=True))
    denominator = _dot(difference, difference)
    numerator = -_dot(second, difference)
    weight = torch.clamp(numerator / (denominator + eps), 0.0, 1.0)
    return tuple(
        weight * left + (1.0 - weight) * right
        for left, right in zip(first, second, strict=True)
    )


def gradient_geometry(raw: Sequence, corrected: Sequence, *, eps: float = 1e-12):
    """Return correction ratio and raw/corrected cosine alignment tensors."""
    import torch

    raw_norm = torch.sqrt(_dot(raw, raw))
    corrected_norm = torch.sqrt(_dot(corrected, corrected))
    delta = tuple(after - before for before, after in zip(raw, corrected, strict=True))
    correction_ratio = torch.sqrt(_dot(delta, delta)) / (raw_norm + eps)
    alignment = _dot(raw, corrected) / (raw_norm * corrected_norm + eps)
    return correction_ratio, alignment


def cosine_between(task_rows: Sequence[Sequence], *, eps: float = 1e-12):
    """Return cosine similarity between exactly two task-gradient rows."""
    if len(task_rows) != 2:
        raise ValueError("task-gradient cosine requires exactly two tasks")
    first, second = task_rows
    return _dot(first, second) / (
        _dot(first, first).sqrt() * _dot(second, second).sqrt() + eps
    )


def assign_gradients(parameters: Sequence, gradients: Sequence) -> None:
    """Copy a combined gradient tuple into optimizer-visible parameter buffers."""
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.detach().clone()
