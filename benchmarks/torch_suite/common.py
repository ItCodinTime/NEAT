"""Shared experiment plumbing for the PyTorch benchmark suite."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.baselines import BASELINES, Lion
from benchmarks.manifest import reproducibility_metadata, write_json

DEFAULT_SEEDS = (7, 11, 19, 23, 29)
STANDARD_OPTIMIZERS = ("neat", "adamw", "sgd", "lion")


def add_seed_arguments(parser) -> None:
    """Add public multi-seed controls and the internal single-trial seed."""
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--allow-single-seed", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)


def _without_seed_controls(arguments: list[str]) -> list[str]:
    cleaned = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--allow-single-seed":
            index += 1
            continue
        if token == "--seed":
            index += 2
            continue
        if token == "--seeds":
            index += 1
            while index < len(arguments) and not arguments[index].startswith("--"):
                index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def orchestrate_seeds(module: str, args, argv=None) -> bool:
    """Launch one isolated process per seed, or authorize an internal trial."""
    if args.seed is not None:
        orchestrated = os.environ.get("NEAT_BENCHMARK_ORCHESTRATED") == "1"
        if not orchestrated and not args.allow_single_seed:
            raise SystemExit("single-seed runs require --allow-single-seed")
        args.preliminary = (
            not orchestrated or os.environ.get("NEAT_BENCHMARK_PRELIMINARY") == "1"
        )
        args.seeds = [args.seed]
        return False
    seeds = list(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    if len(seeds) == 1 and not args.allow_single_seed:
        raise SystemExit("single-seed runs require --allow-single-seed")
    original = list(sys.argv[1:] if argv is None else argv)
    base = _without_seed_controls(original)
    environment = os.environ.copy()
    environment["NEAT_BENCHMARK_ORCHESTRATED"] = "1"
    if len(seeds) == 1:
        environment["NEAT_BENCHMARK_PRELIMINARY"] = "1"
    for seed in seeds:
        subprocess.run(
            [sys.executable, "-m", module, *base, "--seed", str(seed)],
            check=True,
            env=environment,
        )
    return True


def require_torch():
    """Import PyTorch with an actionable benchmark dependency error."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This benchmark requires PyTorch. Install `pip install -e '.[benchmarks]'`."
        ) from exc
    return torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every visible CUDA device."""
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str):
    """Resolve `auto` to CUDA, then MPS, then CPU in priority order."""
    torch = require_torch()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name: str, parameters, lr: float, weight_decay: float):
    """Build a comparator from a shared minimal optimizer interface."""
    torch = require_torch()
    if name not in BASELINES:
        raise ValueError(f"unsupported optimizer: {name}")
    if BASELINES[name].requires_task_gradients:
        raise ValueError(
            f"{name} requires separate task gradients and has no direct "
            "single-objective momentum-opponent analogue"
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    if name == "lion":
        return Lion.build(parameters, lr=lr, weight_decay=weight_decay)
    if name == "neat":
        from neat_optim import TorchNEAT

        return TorchNEAT(
            parameters,
            learning_rate=lr,
            # Sampling avoids turning scalar diagnostic reductions into the
            # dominant cost on MPS while still covering every training phase.
            diagnostic_interval=10,
            weight_decay=weight_decay,
            alpha=0.25,
            beta=0.9,
            opponent_source="previous_gradient",
            adaptive_alpha=True,
            adaptive_preconditioning=True,
            bias_correction=True,
        )
    raise AssertionError(f"registry entry has no builder: {name}")


@dataclass
class EpochMetrics:
    """Metrics persisted for one epoch or RL evaluation checkpoint."""

    epoch: int
    train_loss: float
    eval_metric: float
    seconds: float
    loss_variance: float
    learning_rate: float


class ExperimentLogger:
    """Write JSONL/TensorBoard logs and a self-contained result manifest."""

    def __init__(self, output_dir: str, run_name: str, config: dict[str, Any]):
        """Create an isolated run directory and optional TensorBoard writer."""
        self.path = Path(output_dir) / run_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.config = {"tuning_status": "untuned_shared", **config}
        self.history: list[EpochMetrics] = []
        self.started = time.time()
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(self.path / "tensorboard")
        except ImportError:
            pass
        write_json(self.path / "config.json", self.config)

    def log(self, metrics: EpochMetrics) -> None:
        """Append one durable JSONL row and mirror it to TensorBoard."""
        self.history.append(metrics)
        row = asdict(metrics)
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self._writer:
            for key, value in row.items():
                if key != "epoch":
                    self._writer.add_scalar(key, value, metrics.epoch)

    def finish(
        self,
        *,
        device,
        target: float | None,
        higher_is_better: bool,
        optimizer,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize convergence metadata and write the complete manifest."""
        reached = None
        if target is not None:
            for row in self.history:
                hit = (
                    row.eval_metric >= target
                    if higher_is_better
                    else row.eval_metric <= target
                )
                if hit:
                    reached = row.epoch
                    break
        seed = int(self.config["seed"])
        provenance = reproducibility_metadata(
            seed=seed, device=device, invocation=sys.argv
        )
        result = {
            "schema_version": 3,
            "config": self.config,
            "environment": provenance,
            "reproducibility": provenance,
            "preliminary": bool(self.config.get("preliminary", False))
            or os.environ.get("NEAT_BENCHMARK_PRELIMINARY") == "1",
            "history": [asdict(row) for row in self.history],
            "epochs_to_target": reached,
            "target": target,
            "wall_time_seconds": time.time() - self.started,
            "optimizer_diagnostics": (
                optimizer.diagnostic_snapshot()
                if hasattr(optimizer, "diagnostic_snapshot")
                else None
            ),
            **(extra or {}),
        }
        write_json(self.path / "result.json", result)
        if self._writer:
            self._writer.close()
        print(json.dumps(result, indent=2))
        return result


def run_name(task: str, optimizer: str, seed: int) -> str:
    """Return a sortable, collision-resistant human-readable run name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{task}-{optimizer}-seed{seed}-{stamp}"
