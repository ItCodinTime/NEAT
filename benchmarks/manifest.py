"""Reproducibility metadata shared by every benchmark harness."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: Iterable[str] | None = None) -> dict[str, str]:
    """Return installed versions without importing benchmark frameworks."""
    requested = names or ("neat-optim", "numpy", "torch", "keras", "scipy")
    versions: dict[str, str] = {}
    for name in requested:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def reproducibility_metadata(
    *, seed: int, device: Any = None, invocation: list[str] | None = None
) -> dict[str, Any]:
    """Capture provenance required to reproduce or exclude a run."""
    dirty_output = _git(["status", "--porcelain"])
    metadata: dict[str, Any] = {
        "git_sha": _git(["rev-parse", "HEAD"]) or "unknown",
        "dirty_tree": bool(dirty_output) if dirty_output is not None else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "device": str(device) if device is not None else "unknown",
        },
        "package_versions": package_versions(),
        "rng_seeds": {"python": seed, "numpy": seed, "torch": seed},
        "cli_invocation": list(invocation or sys.argv),
    }
    try:
        import torch

        metadata["hardware"]["torch_device"] = str(device or "unknown")
        if device is not None and getattr(device, "type", None) == "cuda":
            metadata["hardware"]["accelerator"] = torch.cuda.get_device_name(device)
    except ImportError:
        pass
    return metadata


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def validate_paper_eligible(manifest: dict[str, Any]) -> None:
    """Reject preliminary or dirty runs from paper aggregation."""
    if manifest.get("preliminary", False):
        raise ValueError("preliminary single-seed runs are not paper-eligible")
    provenance = manifest.get("reproducibility", manifest.get("environment", {}))
    if provenance.get("dirty_tree") is not False:
        raise ValueError("dirty or unverified worktree is not paper-eligible")


def seed_libraries(seed: int) -> None:
    """Seed libraries available in the current process."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
