"""Recorded, symmetric hyperparameter-grid selection for benchmark baselines."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmarks.manifest import write_json


def grid(config: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand a named Cartesian grid in deterministic key/value order."""
    keys = sorted(config)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(config[key] for key in keys))
    ]


def run_tuning(
    *,
    task: str,
    optimizers: Iterable[str],
    search_space: Mapping[str, Iterable[Any]],
    evaluate: Callable[[str, dict[str, Any]], float],
    output_dir: str | Path,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Evaluate the identical grid for each optimizer and record every trial."""
    candidates = grid(search_space)
    names = list(optimizers)
    if not names or not candidates:
        raise ValueError("tuning requires optimizers and a non-empty search space")
    trials = []
    selected: dict[str, dict[str, Any]] = {}
    for optimizer in names:
        optimizer_trials = []
        for configuration in candidates:
            metric = float(evaluate(optimizer, configuration.copy()))
            row = {
                "optimizer": optimizer,
                "config": configuration,
                "validation_metric": metric,
            }
            trials.append(row)
            optimizer_trials.append(row)

        def key(row: dict[str, Any]) -> float:
            return float(row["validation_metric"])

        selected[optimizer] = dict(
            (max if higher_is_better else min)(optimizer_trials, key=key)
        )
    report = {
        "schema_version": 1,
        "task": task,
        "optimizers": names,
        "search_space": {key: list(value) for key, value in search_space.items()},
        "trials": trials,
        "selected": selected,
    }
    destination = Path(output_dir)
    write_json(destination / "tuning_manifest.json", report)
    lines = [
        f"# Tuning report: {task}",
        "",
        "| Optimizer | Configuration | Validation metric |",
        "|---|---|---:|",
    ]
    for row in trials:
        lines.append(
            f"| {row['optimizer']} | `{row['config']}` | {row['validation_metric']} |"
        )
    (destination / "tuning_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def require_all_optimizers_tuned(
    tuning_manifest: Mapping[str, Any], compared_optimizers: Iterable[str]
) -> None:
    """Prevent a comparison from mixing tuned and untuned optimizers."""
    selected = set(tuning_manifest.get("selected", {}))
    compared = set(compared_optimizers)
    if selected != compared:
        raise ValueError(
            f"tuning mismatch: tuned={sorted(selected)}, compared={sorted(compared)}"
        )
