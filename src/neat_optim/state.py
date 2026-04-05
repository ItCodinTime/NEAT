"""State containers shared by the reference and functional engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ArrayState:
    """Per-parameter optimizer state for the reference engine."""

    momentum: np.ndarray
    nce: np.ndarray
    previous_gradient: np.ndarray | None = None
    gradient_ema: np.ndarray | None = None
    step: int = 0

    @classmethod
    def zeros_like(cls, array: np.ndarray) -> ArrayState:
        """Create a zero-initialized state matching `array`."""
        return cls(
            momentum=np.zeros_like(array, dtype=np.float32),
            nce=np.zeros_like(array, dtype=np.float32),
            previous_gradient=np.zeros_like(array, dtype=np.float32),
            gradient_ema=np.zeros_like(array, dtype=np.float32),
            step=0,
        )


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Scalar diagnostics produced by a single NEAT step."""

    grad_norm: float
    update_norm: float
    nce_norm: float
    conflict_ratio: float
    active_fraction: float = 1.0
    opponent_norm: float = 0.0
    correction_ratio: float = 0.0
    update_alignment: float = 1.0
    correction_active: float = 0.0


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result bundle for a single functional or reference-engine step."""

    param: np.ndarray
    state: ArrayState
    metrics: StepMetrics


@dataclass(frozen=True, slots=True)
class PlayerStepMetrics:
    """Diagnostics produced by a player-aware NEAT step."""

    grad_norm: float
    update_norm: float
    nce_norm: float
    mean_player_conflict: float
    max_player_conflict: float
    active_fraction: float
    num_players: int
    mean_correction_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class PlayerStepResult:
    """Result bundle for a single player-aware NEAT step."""

    param: np.ndarray
    state: ArrayState
    metrics: PlayerStepMetrics
