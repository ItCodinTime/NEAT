from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_conflicting_gradient(step: int, size: int = 128) -> np.ndarray:
    base = np.ones((size,), dtype=np.float32)
    return base if step % 2 == 0 else -base


@dataclass(frozen=True)
class QuadraticConflictProblem:
    """Deterministic two-player quadratic with an analytic mean-loss optimum."""

    centers: np.ndarray
    curvature: np.ndarray

    @classmethod
    def create(
        cls, *, dimensionality: int, conflict_intensity: float
    ) -> QuadraticConflictProblem:
        if dimensionality < 1:
            raise ValueError("dimensionality must be positive")
        if not 0.0 <= conflict_intensity <= 1.0:
            raise ValueError("conflict_intensity must be in [0, 1]")
        direction = np.ones(dimensionality, dtype=np.float64)
        first = direction
        second = (1.0 - 2.0 * conflict_intensity) * direction
        return cls(
            centers=np.stack((first, second)),
            curvature=np.ones((2, dimensionality), dtype=np.float64),
        )

    @property
    def optimum(self) -> np.ndarray:
        return np.sum(self.curvature * self.centers, axis=0) / np.sum(
            self.curvature, axis=0
        )

    def gradients(
        self, parameters: np.ndarray, *, noise_level: float = 0.0, seed: int = 0
    ) -> np.ndarray:
        if noise_level < 0.0:
            raise ValueError("noise_level must be non-negative")
        gradients = self.curvature * (parameters - self.centers)
        if noise_level:
            rng = np.random.default_rng(seed)
            gradients = gradients + rng.normal(scale=noise_level, size=gradients.shape)
        return gradients.astype(np.float32)

    def losses(self, parameters: np.ndarray) -> np.ndarray:
        displacement = parameters - self.centers
        return 0.5 * np.sum(self.curvature * displacement**2, axis=1)
