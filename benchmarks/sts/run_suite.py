"""Run resumable STS experiment matrices from locked configurations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.manifest import write_json
from benchmarks.sts import DATASETS, METHODS, PAPER_SEEDS
from benchmarks.sts.multitask_vision import NEAT_VARIANTS, TASK_NEAT_VARIANTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("smoke", "tier1", "mechanism", "ablations", "promotion")
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", default="data/sts")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--locked-lrs", default="benchmark-runs/sts/locked_lrs.json")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            "benchmark-runs/sts-v2-gate"
            if args.phase == "promotion"
            else "benchmark-runs/sts"
        )
    return args


def _locked_lrs(path: str) -> dict[str, dict[str, float]]:
    source = Path(path)
    if not source.exists():
        raise SystemExit(
            f"locked learning rates not found: {source}; run benchmarks.sts.tune first"
        )
    return json.loads(source.read_text(encoding="utf-8"))["selected"]


def _matrix(args, learning_rates):
    if args.phase == "smoke":
        for dataset in args.datasets:
            for method in METHODS:
                yield dataset, method, "multi", "base", args.seeds[0], 1, 512, 256
    elif args.phase == "tier1":
        for dataset in args.datasets:
            for method in METHODS:
                for seed in args.seeds:
                    yield (
                        dataset,
                        method,
                        "multi",
                        "base",
                        seed,
                        args.epochs,
                        None,
                        None,
                    )
    elif args.phase == "mechanism":
        for dataset in args.datasets:
            for setting in ("single_task_1", "single_task_2"):
                for seed in args.seeds:
                    yield (
                        dataset,
                        "neat",
                        setting,
                        "base",
                        seed,
                        args.epochs,
                        None,
                        None,
                    )
    elif args.phase == "ablations":
        variants = [variant for variant in NEAT_VARIANTS if variant != "base"]
        for dataset in args.datasets:
            for variant in variants:
                for seed in args.seeds:
                    yield (
                        dataset,
                        "neat",
                        "multi",
                        variant,
                        seed,
                        args.epochs,
                        None,
                        None,
                    )
    else:
        variants = [
            variant for variant in TASK_NEAT_VARIANTS if variant.startswith("adam_")
        ]
        for dataset in args.datasets:
            for seed in args.seeds:
                yield dataset, "adam", "multi", "base", seed, args.epochs, None, None
                for variant in variants:
                    yield (
                        dataset,
                        "task_neat",
                        "multi",
                        variant,
                        seed,
                        args.epochs,
                        None,
                        None,
                    )


def main(argv=None) -> None:
    args = parse_args(argv)
    learning_rates = (
        {dataset: {method: 1e-3 for method in METHODS} for dataset in args.datasets}
        if args.phase == "smoke"
        else _locked_lrs(args.locked_lrs)
    )
    commands = []
    for (
        dataset,
        method,
        setting,
        variant,
        seed,
        epochs,
        train_limit,
        eval_limit,
    ) in _matrix(args, learning_rates):
        command = [
            sys.executable,
            "-m",
            "benchmarks.sts.multitask_vision",
            "--dataset",
            dataset,
            "--method",
            method,
            "--setting",
            setting,
            "--neat-variant",
            variant,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--lr",
            str(learning_rates[dataset].get(method, learning_rates[dataset]["adam"])),
            "--data-dir",
            args.data_dir,
            "--output-dir",
            args.output_dir,
        ]
        if train_limit is not None:
            command.extend(["--train-limit", str(train_limit)])
        if eval_limit is not None:
            command.extend(["--eval-limit", str(eval_limit)])
        if args.phase == "promotion":
            command.append("--skip-test")
        commands.append(command)
    manifest = {
        "schema_version": 1,
        "phase": args.phase,
        "seeds": args.seeds,
        "learning_rates": learning_rates,
        "test_split_used": args.phase != "promotion",
        "commands": commands,
    }
    write_json(Path(args.output_dir) / "manifests" / f"{args.phase}.json", manifest)
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
