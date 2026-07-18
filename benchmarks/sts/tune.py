"""Tune learning rates on validation accuracy without reading the test split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.manifest import write_json
from benchmarks.sts import DATASETS, METHODS
from benchmarks.sts.multitask_vision import parse_args as parse_trial_args
from benchmarks.sts.multitask_vision import result_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3]
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data/sts")
    parser.add_argument("--output-dir", default="benchmark-runs/sts/tuning")
    parser.add_argument("--locked-output", default="benchmark-runs/sts/locked_lrs.json")
    return parser.parse_args(argv)


def _trial_path(args, dataset: str, method: str, seed: int, lr: float) -> Path:
    output_dir = Path(args.output_dir) / f"lr_{lr:g}"
    trial = parse_trial_args(
        [
            "--dataset",
            dataset,
            "--method",
            method,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]
    )
    return result_path(trial)


def main(argv=None) -> None:
    args = parse_args(argv)
    trials = []
    for dataset in args.datasets:
        for method in args.methods:
            for lr in args.learning_rates:
                for seed in args.seeds:
                    command = [
                        sys.executable,
                        "-m",
                        "benchmarks.sts.multitask_vision",
                        "--dataset",
                        dataset,
                        "--method",
                        method,
                        "--setting",
                        "multi",
                        "--seed",
                        str(seed),
                        "--lr",
                        str(lr),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--workers",
                        str(args.workers),
                        "--data-dir",
                        args.data_dir,
                        "--output-dir",
                        str(Path(args.output_dir) / f"lr_{lr:g}"),
                        "--skip-test",
                    ]
                    subprocess.run(command, check=True)
                    path = _trial_path(args, dataset, method, seed, lr)
                    result = json.loads(path.read_text(encoding="utf-8"))
                    tail = result["history"][-min(3, args.epochs) :]
                    score = float(
                        np.mean([row["validation_average_accuracy"] for row in tail])
                    )
                    trials.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "learning_rate": lr,
                            "validation_score": score,
                            "result": str(path),
                            "command": command,
                        }
                    )
    selected = {}
    for dataset in args.datasets:
        selected[dataset] = {}
        for method in args.methods:
            candidates = []
            for lr in args.learning_rates:
                scores = [
                    row["validation_score"]
                    for row in trials
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["learning_rate"] == lr
                ]
                candidates.append((float(np.mean(scores)), -lr, lr))
            selected[dataset][method] = max(candidates)[2]
    report = {
        "schema_version": 1,
        "selection_metric": "mean final-three-epoch validation average accuracy",
        "test_split_used": False,
        "seeds": args.seeds,
        "learning_rates": args.learning_rates,
        "epochs": args.epochs,
        "selected": selected,
        "trials": trials,
    }
    write_json(args.locked_output, report)
    write_json(Path(args.output_dir) / "tuning_manifest.json", report)
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
