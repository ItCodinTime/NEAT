"""Shared experiment plumbing for the PyTorch benchmark suite."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
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
    raise ValueError(f"unsupported optimizer: {name}")


def git_commit() -> str:
    """Return the checked-out revision, or `unknown` outside a Git tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata(device) -> dict[str, Any]:
    """Capture enough runtime metadata to interpret benchmark timings."""
    torch = require_torch()
    metadata = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "hostname": platform.node(),
    }
    if device.type == "cuda":
        metadata["accelerator"] = torch.cuda.get_device_name(device)
    return metadata


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
        self.config = config
        self.history: list[EpochMetrics] = []
        self.started = time.time()
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(self.path / "tensorboard")
        except ImportError:
            pass
        (self.path / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

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
        result = {
            "config": self.config,
            "environment": environment_metadata(device),
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
        (self.path / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if self._writer:
            self._writer.close()
        print(json.dumps(result, indent=2))
        return result


def run_name(task: str, optimizer: str, seed: int) -> str:
    """Return a sortable, collision-resistant human-readable run name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{task}-{optimizer}-seed{seed}-{stamp}"
