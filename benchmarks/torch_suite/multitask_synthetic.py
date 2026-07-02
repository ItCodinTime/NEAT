"""Run task-gradient baselines on the deterministic quadratic testbed."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.baselines import cagrad, pcgrad
from benchmarks.manifest import reproducibility_metadata, write_json
from benchmarks.tasks.synthetic_conflict import QuadraticConflictProblem
from benchmarks.torch_suite.common import (
    add_seed_arguments,
    orchestrate_seeds,
    run_name,
)
from neat_optim import ArrayState, PlayerNEATConfig
from neat_optim.engine.multiplayer import neat_player_step

TASK_GRADIENT_OPTIMIZERS = ("neat", "pcgrad", "cagrad", "sgd")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=TASK_GRADIENT_OPTIMIZERS, default="neat")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dimensionality", type=int, default=32)
    parser.add_argument("--conflict-intensity", type=float, default=0.75)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output-dir", default="benchmark-runs/multitask-synthetic")
    add_seed_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if orchestrate_seeds("benchmarks.torch_suite.multitask_synthetic", args, argv):
        return
    started_run = time.perf_counter()
    problem = QuadraticConflictProblem.create(
        dimensionality=args.dimensionality,
        conflict_intensity=args.conflict_intensity,
    )
    parameter = np.zeros(args.dimensionality, dtype=np.float32)
    state = ArrayState.zeros_like(parameter)
    momentum = np.zeros_like(parameter)
    history = []
    for step in range(args.steps):
        started = time.perf_counter()
        task_gradients = problem.gradients(
            parameter,
            noise_level=args.noise_level,
            seed=args.seed * 1_000_003 + step,
        )
        correction_ratio = 0.0
        conflict = 0.0
        if args.optimizer == "neat":
            result = neat_player_step(
                parameter,
                task_gradients,
                state,
                PlayerNEATConfig(
                    learning_rate=args.lr,
                    beta=args.momentum,
                    native="never",
                ),
            )
            parameter, state = result.param, result.state
            correction_ratio = result.metrics.mean_correction_ratio
            conflict = result.metrics.mean_player_conflict
        else:
            if args.optimizer == "pcgrad":
                combined = pcgrad(task_gradients, seed=args.seed + step)
            elif args.optimizer == "cagrad":
                combined = cagrad(task_gradients)
            else:
                combined = task_gradients.mean(axis=0)
            momentum = args.momentum * momentum + (1.0 - args.momentum) * combined
            parameter = parameter - args.lr * momentum
        losses = problem.losses(parameter)
        history.append(
            {
                "epoch": step + 1,
                "train_loss": float(losses.mean()),
                "eval_metric": float(np.linalg.norm(parameter - problem.optimum)),
                "seconds": time.perf_counter() - started,
                "loss_variance": float(np.var(losses)),
                "learning_rate": args.lr,
                "conflict_coefficient": conflict,
                "correction_ratio": correction_ratio,
            }
        )
    manifest = {
        "schema_version": 3,
        "config": {
            **vars(args),
            "task": "multitask_synthetic",
            "tuning_status": "untuned_shared",
        },
        "history": history,
        "wall_time_seconds": time.perf_counter() - started_run,
        "preliminary": args.preliminary,
        "reproducibility": reproducibility_metadata(seed=args.seed),
        "known_optimum": problem.optimum.tolist(),
        "final_parameter": parameter.tolist(),
        "metric": "distance_to_known_optimum",
    }
    destination = (
        Path(args.output_dir)
        / run_name("multitask-synthetic", args.optimizer, args.seed)
        / "result.json"
    )
    write_json(destination, manifest)


if __name__ == "__main__":
    main()
