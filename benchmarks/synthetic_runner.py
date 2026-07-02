"""Deterministic reference-engine experiments for mechanisms and regime maps."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from benchmarks.tasks.synthetic_conflict import QuadraticConflictProblem
from neat_optim import ArrayState, NEATConfig
from neat_optim.engine.reference import neat_step_reference


def run_synthetic(
    *,
    config: NEATConfig,
    seed: int,
    steps: int,
    dimensionality: int,
    conflict_intensity: float,
    noise_level: float,
    batch_size: int,
) -> dict[str, Any]:
    """Run one fully specified quadratic trial and return per-step traces."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    problem = QuadraticConflictProblem.create(
        dimensionality=dimensionality,
        conflict_intensity=conflict_intensity,
    )
    parameter = np.zeros(dimensionality, dtype=np.float32)
    state = ArrayState.zeros_like(parameter)
    history = []
    for step in range(steps):
        started = time.perf_counter()
        batch_gradients = [
            problem.gradients(
                parameter,
                noise_level=noise_level,
                seed=seed * 1_000_003 + step * batch_size + sample,
            ).mean(axis=0)
            for sample in range(batch_size)
        ]
        gradient = np.mean(batch_gradients, axis=0).astype(np.float32)
        result = neat_step_reference(parameter, gradient, state, config)
        parameter, state = result.param, result.state
        history.append(
            {
                "epoch": step + 1,
                "train_loss": float(problem.losses(parameter).mean()),
                "eval_metric": float(np.linalg.norm(parameter - problem.optimum)),
                "seconds": time.perf_counter() - started,
                "loss_variance": float(np.var(problem.losses(parameter))),
                "learning_rate": config.learning_rate,
                "conflict_coefficient": result.metrics.conflict_ratio,
                "correction_ratio": result.metrics.correction_ratio,
                "alignment_ema": state.alignment_ema,
                "gate_active": result.metrics.correction_active,
            }
        )
    return {
        "config": config.as_dict(),
        "seed": seed,
        "history": history,
        "final_distance_to_optimum": history[-1]["eval_metric"],
        "known_optimum": problem.optimum.tolist(),
        "final_parameter": parameter.tolist(),
        "final_state": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(state).items()
        },
    }
