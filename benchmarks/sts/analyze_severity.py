"""Analyze the validation-only task-conflict severity sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", nargs="?", default="benchmark-runs/sts-conflict-severity"
    )
    return parser.parse_args(argv)


def _load_rows(run_dir: Path) -> list[dict[str, float | int | str]]:
    indexed = {}
    for path in run_dir.glob("runs/**/result.json"):
        result = json.loads(path.read_text())
        config = result["config"]
        if result.get("test") is not None:
            raise ValueError(f"severity run used the test split: {path}")
        history = result["history"]
        key = (
            config["dataset"],
            int(config["overlap_offset"]),
            int(config["seed"]),
            config["neat_variant"],
        )
        indexed[key] = {
            "validation_accuracy": float(history[-1]["validation_average_accuracy"]),
            "task_gradient_cosine": float(
                np.mean([row["task_gradient_cosine"] for row in history])
            ),
            "correction_ratio": float(
                np.mean([row["correction_ratio"] for row in history])
            ),
        }

    rows = []
    groups = {(dataset, offset, seed) for dataset, offset, seed, _ in indexed}
    for dataset, offset, seed in sorted(groups):
        control = indexed[(dataset, offset, seed, "adam_alpha_0")]
        candidate = indexed[(dataset, offset, seed, "adam_alpha_0_5")]
        rows.append(
            {
                "dataset": dataset,
                "overlap_offset": offset,
                "overlap_pixels_per_axis": 28 - offset,
                "seed": seed,
                "task_gradient_cosine": candidate["task_gradient_cosine"],
                "correction_ratio": candidate["correction_ratio"],
                "control_validation_accuracy": control["validation_accuracy"],
                "candidate_validation_accuracy": candidate["validation_accuracy"],
                "paired_validation_gain": candidate["validation_accuracy"]
                - control["validation_accuracy"],
            }
        )
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    rows = _load_rows(run_dir)
    if not rows:
        raise SystemExit(f"no completed severity runs found under {run_dir}")
    output = run_dir / "severity_assets"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "severity_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["overlap_offset"])].append(row)
    summary = []
    for (dataset, offset), values in sorted(grouped.items()):
        gains = np.asarray([row["paired_validation_gain"] for row in values])
        summary.append(
            {
                "dataset": dataset,
                "overlap_offset": offset,
                "overlap_pixels_per_axis": 28 - offset,
                "seed_count": len(values),
                "mean_task_gradient_cosine": float(
                    np.mean([row["task_gradient_cosine"] for row in values])
                ),
                "mean_correction_ratio": float(
                    np.mean([row["correction_ratio"] for row in values])
                ),
                "mean_paired_validation_gain": float(gains.mean()),
                "std_paired_validation_gain": float(gains.std(ddof=1))
                if len(gains) > 1
                else 0.0,
            }
        )
    (output / "severity_summary.json").write_text(
        json.dumps(
            {"schema_version": 1, "test_split_used": False, "rows": summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dataset in sorted({str(row["dataset"]) for row in summary}):
        points = [row for row in summary if row["dataset"] == dataset]
        axes[0].plot(
            [row["mean_task_gradient_cosine"] for row in points],
            [row["mean_paired_validation_gain"] for row in points],
            marker="o",
            label=dataset,
        )
        axes[1].plot(
            [row["mean_correction_ratio"] for row in points],
            [row["mean_paired_validation_gain"] for row in points],
            marker="o",
            label=dataset,
        )
    axes[0].set(xlabel="Mean task-gradient cosine", ylabel="Paired validation gain")
    axes[1].set(xlabel="Mean correction ratio", ylabel="Paired validation gain")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "severity_response.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
