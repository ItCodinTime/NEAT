"""Baseline registry and framework-free multi-objective gradient surgery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Baseline:
    name: str
    requires_task_gradients: bool = False
    applies_to_single_objective: bool = True


BASELINES = {
    baseline.name: baseline
    for baseline in (
        Baseline("neat"),
        Baseline("adamw"),
        Baseline("sgd"),
        Baseline("lion"),
        Baseline(
            "pcgrad", requires_task_gradients=True, applies_to_single_objective=False
        ),
        Baseline(
            "cagrad", requires_task_gradients=True, applies_to_single_objective=False
        ),
    )
}


def pcgrad(
    task_gradients: np.ndarray, *, seed: int = 0, eps: float = 1e-12
) -> np.ndarray:
    """Apply Algorithm 1 of PCGrad to a ``[tasks, parameters]`` matrix."""
    gradients = np.asarray(task_gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("PCGrad requires at least two task-gradient rows")
    projected = gradients.copy()
    rng = np.random.default_rng(seed)
    for task in range(len(gradients)):
        for other in rng.permutation(len(gradients)):
            if other == task:
                continue
            denominator = float(np.dot(gradients[other], gradients[other]))
            inner = float(np.dot(projected[task], gradients[other]))
            if inner < 0.0 and denominator > eps:
                projected[task] -= (inner / denominator) * gradients[other]
    return np.mean(projected, axis=0).astype(task_gradients.dtype, copy=False)


def cagrad(
    task_gradients: np.ndarray,
    *,
    conflict_aversion: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Solve the CAGrad simplex dual and return its shared update direction."""
    gradients = np.asarray(task_gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        raise ValueError("CAGrad requires at least two task-gradient rows")
    if conflict_aversion < 0:
        raise ValueError("conflict_aversion must be non-negative")
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("CAGrad requires the benchmark extra (scipy)") from exc
    task_count = gradients.shape[0]
    gram = gradients @ gradients.T
    uniform = np.full(task_count, 1.0 / task_count)
    mean_gradient = np.mean(gradients, axis=0)
    coefficient = conflict_aversion * np.linalg.norm(mean_gradient)

    def objective(weights: np.ndarray) -> float:
        weighted_norm = np.sqrt(max(float(weights @ gram @ weights), 0.0) + eps)
        return float(weights @ gram @ uniform) + coefficient * weighted_norm

    result = minimize(
        objective,
        uniform,
        bounds=[(0.0, 1.0)] * task_count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        method="SLSQP",
    )
    if not result.success:
        raise RuntimeError(f"CAGrad dual solver failed: {result.message}")
    weighted = result.x @ gradients
    multiplier = coefficient / (np.linalg.norm(weighted) + eps)
    combined = (mean_gradient + multiplier * weighted) / (1.0 + conflict_aversion)
    return combined.astype(task_gradients.dtype, copy=False)


class Lion:
    """Dependency-local factory for the Lion optimizer in PyTorch."""

    @staticmethod
    def build(parameters: Any, *, lr: float, weight_decay: float):
        import torch

        class _Lion(torch.optim.Optimizer):
            def __init__(self, params):
                super().__init__(
                    params,
                    {"lr": lr, "betas": (0.9, 0.99), "weight_decay": weight_decay},
                )

            @torch.no_grad()
            def step(self, closure=None):
                loss = None if closure is None else closure()
                for group in self.param_groups:
                    beta1, beta2 = group["betas"]
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            continue
                        gradient = parameter.grad
                        state = self.state[parameter]
                        momentum = state.setdefault(
                            "momentum", torch.zeros_like(parameter)
                        )
                        if group["weight_decay"]:
                            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                        update = momentum.mul(beta1).add(gradient, alpha=1.0 - beta1)
                        parameter.add_(update.sign(), alpha=-group["lr"])
                        momentum.mul_(beta2).add_(gradient, alpha=1.0 - beta2)
                return loss

        return _Lion(parameters)
