from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ArrayState:
    """Per-array optimizer state for the reference engine."""

    momentum: np.ndarray
    nce: np.ndarray
    step: int = 0

    @classmethod
    def zeros_like(cls, array: np.ndarray) -> ArrayState:
        return cls(
            momentum=np.zeros_like(array, dtype=np.float32),
            nce=np.zeros_like(array, dtype=np.float32),
            step=0,
        )


@dataclass(frozen=True, slots=True)
class StepMetrics:
    grad_norm: float
    update_norm: float
    nce_norm: float
    conflict_ratio: float


@dataclass(frozen=True, slots=True)
class StepResult:
    param: np.ndarray
    state: ArrayState
    metrics: StepMetrics
