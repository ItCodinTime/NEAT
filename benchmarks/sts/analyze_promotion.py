"""Create a validation-only decision report for the task-aware NEAT v2 gate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from benchmarks.manifest import write_json
from benchmarks.sts import DATASETS

CONTROL = "adam_alpha_0"
CANDIDATES = ("adam_alpha_0_25", "adam_alpha_0_5", "adam_alpha_1")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument("--collapse-margin", type=float, default=0.05)
    return parser.parse_args(argv)


def _result_path(root: Path, dataset: str, variant: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / dataset
        / "multi"
        / "task_neat"
        / variant
        / f"seed_{seed}"
        / "result.json"
    )


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing promotion result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("test") is not None:
        raise SystemExit(f"promotion result accessed the test split: {path}")
    if not result.get("history"):
        raise SystemExit(f"promotion result has no epoch history: {path}")
    return result


def _score(result: dict) -> float:
    tail = result["history"][-3:]
    return statistics.mean(row["validation_average_accuracy"] for row in tail)


def _correction(result: dict) -> float:
    return statistics.mean(row["correction_ratio"] for row in result["history"])


def main(argv=None) -> None:
    args = parse_args(argv)
    rows = []
    decisions = {}
    variants = (CONTROL, *CANDIDATES)
    for dataset in args.datasets:
        scores = {}
        corrections = {}
        for variant in variants:
            results = [
                _load(_result_path(args.run_dir, dataset, variant, seed))
                for seed in args.seeds
            ]
            scores[variant] = [_score(result) for result in results]
            corrections[variant] = statistics.mean(
                _correction(result) for result in results
            )
        control_scores = scores[CONTROL]
        for variant in CANDIDATES:
            paired = [
                candidate - control
                for candidate, control in zip(
                    scores[variant], control_scores, strict=True
                )
            ]
            collapsed = any(
                candidate < control - args.collapse_margin
                for candidate, control in zip(
                    scores[variant], control_scores, strict=True
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "control_mean": statistics.mean(control_scores),
                    "candidate_mean": statistics.mean(scores[variant]),
                    "mean_paired_gain": statistics.mean(paired),
                    "minimum_paired_gain": min(paired),
                    "mean_correction_ratio": corrections[variant],
                    "collapsed": collapsed,
                }
            )
    for variant in CANDIDATES:
        relevant = [row for row in rows if row["variant"] == variant]
        decisions[variant] = {
            "advance": all(
                row["mean_paired_gain"] > 0.0
                and row["mean_correction_ratio"] > 0.0
                and not row["collapsed"]
                for row in relevant
            ),
            "criterion": (
                "positive mean paired validation gain on every dataset, non-zero "
                "correction, and no seed more than collapse_margin below control"
            ),
        }

    destination = args.run_dir / "promotion_assets"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "promotion_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "selection_metric": "mean final-three-epoch validation average accuracy",
        "test_split_used": False,
        "seeds": args.seeds,
        "collapse_margin": args.collapse_margin,
        "rows": rows,
        "decisions": decisions,
    }
    write_json(destination / "promotion_decision.json", report)
    lines = [
        "# Task-aware NEAT v2 promotion gate",
        "",
        "The test split was not used. Results are exploratory validation evidence.",
        "",
        "| Dataset | Variant | Control | Candidate | Paired gain | "
        "Correction | Collapse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | "
            f"{row['control_mean']:.4f} | {row['candidate_mean']:.4f} | "
            f"{row['mean_paired_gain']:+.4f} | "
            f"{row['mean_correction_ratio']:.4f} | {row['collapsed']} |"
        )
    lines.extend(["", "## Decision", ""])
    for variant, decision in decisions.items():
        action = "ADVANCE" if decision["advance"] else "STOP"
        lines.append(f"- `{variant}`: **{action}**")
    (destination / "promotion_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
