"""Tune fairly on validation, lock settings, then run a held-out test study.

This runner is intentionally limited to the CIFAR studies that have a real
validation/test separation.  It gives every comparator the same number of
learning-rate candidates, stores every trial, freezes the selected learning
rate in ``tuning_manifest.json``, and only then starts multi-seed test runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from benchmarks.manifest import write_json
from benchmarks.torch_suite.common import STANDARD_OPTIMIZERS

STUDIES: dict[str, dict[str, Any]] = {
    "cifar10-resnet18": {
        "dataset": "cifar10",
        "model": "resnet18",
        "weight_decay": 5e-4,
        "batch_size": 128,
        "tuning_epochs": 15,
        "test_epochs": 100,
        "validation_fraction": 0.1,
        "target_accuracy": 0.92,
        "learning_rates": {
            "neat": (1e-4, 3e-4, 1e-3, 3e-3),
            "adamw": (1e-4, 3e-4, 1e-3, 3e-3),
            "lion": (1e-5, 3e-5, 1e-4, 3e-4),
            "sgd": (1e-2, 3e-2, 1e-1, 3e-1),
        },
    },
    "cifar100-resnet34": {
        "dataset": "cifar100",
        "model": "resnet34",
        "weight_decay": 5e-4,
        "batch_size": 128,
        "tuning_epochs": 12,
        "test_epochs": 100,
        "validation_fraction": 0.1,
        "target_accuracy": 0.70,
        "learning_rates": {
            "neat": (1e-4, 3e-4, 1e-3, 3e-3),
            "adamw": (1e-4, 3e-4, 1e-3, 3e-3),
            "lion": (1e-5, 3e-5, 1e-4, 3e-4),
            "sgd": (1e-2, 3e-2, 1e-1, 3e-1),
        },
    },
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the fixed protocol and its bounded execution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(STUDIES), required=True)
    parser.add_argument("--phase", choices=("tune", "evaluate", "all"), default="all")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--tuning-seed", type=int, default=101)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 29])
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def _read_result(directory: Path) -> dict[str, Any]:
    results = list(directory.rglob("result.json"))
    if len(results) != 1:
        raise RuntimeError(f"expected one result in {directory}, found {len(results)}")
    return json.loads(results[0].read_text(encoding="utf-8"))


def _has_completed_result(directory: Path) -> bool:
    """Return whether a trial directory already contains exactly one artifact."""
    return len(list(directory.rglob("result.json"))) == 1


def _run_many(
    commands: list[tuple[list[str], Path]],
    args: argparse.Namespace,
    *,
    paper_eligible: bool,
) -> None:
    """Run independent trials concurrently without hiding a failed subprocess."""
    environment = os.environ.copy()
    if paper_eligible:
        environment["NEAT_BENCHMARK_ORCHESTRATED"] = "1"
    if args.torch_threads is not None:
        environment["OMP_NUM_THREADS"] = str(args.torch_threads)
        environment["MKL_NUM_THREADS"] = str(args.torch_threads)

    def run(command: list[str], log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )

    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        pending = {pool.submit(run, command, log): command for command, log in commands}
        for future in as_completed(pending):
            future.result()


def _vision_command(
    *,
    study: str,
    spec: dict[str, Any],
    optimizer: str,
    learning_rate: float,
    epochs: int,
    seed: int,
    output_dir: Path,
    evaluation_split: str,
    tuning_status: str,
    args: argparse.Namespace,
    selection_manifest: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.torch_suite.vision",
        "--dataset",
        str(spec["dataset"]),
        "--model",
        str(spec["model"]),
        "--optimizer",
        optimizer,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(spec["batch_size"]),
        "--lr",
        str(learning_rate),
        "--weight-decay",
        str(spec["weight_decay"]),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--validation-fraction",
        str(spec["validation_fraction"]),
        "--split-seed",
        str(args.split_seed),
        "--evaluation-split",
        evaluation_split,
        "--tuning-status",
        tuning_status,
                        "--study-id",
                        study,
    ]
    if evaluation_split == "test":
        command.extend(["--target-accuracy", str(spec["target_accuracy"])])
    if selection_manifest is not None:
        command.extend(["--selection-manifest", str(selection_manifest)])
    if args.amp:
        command.append("--amp")
    return command


def tune(args: argparse.Namespace, root: Path, spec: dict[str, Any]) -> Path:
    """Run the equal-budget validation search and persist selected settings."""
    commands: list[tuple[list[str], Path]] = []
    for optimizer in STANDARD_OPTIMIZERS:
        for learning_rate in spec["learning_rates"][optimizer]:
            run_dir = root / "tuning" / optimizer / f"lr-{learning_rate:g}"
            commands.append(
                (
                    _vision_command(
                        study=args.study,
                        spec=spec,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        epochs=spec["tuning_epochs"],
                        seed=args.tuning_seed,
                        output_dir=run_dir,
                        evaluation_split="validation",
                        tuning_status="validation_search",
                        args=args,
                    ),
                    run_dir / "trial.log",
                )
            )
    # Tuning uses one reserved seed and cannot be promoted as test evidence.
    for command, _ in commands:
        command.append("--allow-single-seed")
    _run_many(commands, args, paper_eligible=False)

    trials = []
    selected = {}
    for optimizer in STANDARD_OPTIMIZERS:
        optimizer_trials = []
        for learning_rate in spec["learning_rates"][optimizer]:
            run_dir = root / "tuning" / optimizer / f"lr-{learning_rate:g}"
            result = _read_result(run_dir)
            trial = {
                "optimizer": optimizer,
                "config": {"lr": learning_rate},
                "validation_metric": result["history"][-1]["eval_metric"],
                "result": str(next(run_dir.rglob("result.json")).relative_to(root)),
            }
            optimizer_trials.append(trial)
            trials.append(trial)
        selected[optimizer] = max(
            optimizer_trials, key=lambda trial: float(trial["validation_metric"])
        )
    manifest = {
        "schema_version": 1,
        "study": args.study,
        "phase": "validation_search",
        "tuning_seed": args.tuning_seed,
        "split_seed": args.split_seed,
        "validation_fraction": spec["validation_fraction"],
        "epochs": spec["tuning_epochs"],
        "optimizers": list(STANDARD_OPTIMIZERS),
        "learning_rate_candidates": spec["learning_rates"],
        "trials": trials,
        "selected": selected,
    }
    path = root / "tuning_manifest.json"
    write_json(path, manifest)
    return path


def evaluate(
    args: argparse.Namespace, root: Path, spec: dict[str, Any], manifest_path: Path
) -> None:
    """Evaluate only frozen settings on the held-out CIFAR test split."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = manifest.get("selected", {})
    if set(selected) != set(STANDARD_OPTIMIZERS):
        raise ValueError("tuning manifest does not select every compared optimizer")
    commands: list[tuple[list[str], Path]] = []
    for optimizer in STANDARD_OPTIMIZERS:
        learning_rate = float(selected[optimizer]["config"]["lr"])
        for seed in args.seeds:
            run_dir = root / "evaluation" / optimizer / f"seed-{seed}"
            if _has_completed_result(run_dir):
                continue
            commands.append(
                (
                    _vision_command(
                        study=args.study,
                        spec=spec,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        epochs=spec["test_epochs"],
                        seed=seed,
                        output_dir=run_dir,
                        evaluation_split="test",
                        tuning_status="validation_selected",
                        args=args,
                        selection_manifest=manifest_path,
                    ),
                    run_dir / "trial.log",
                )
            )
    _run_many(commands, args, paper_eligible=True)
    results = [
        str(path.relative_to(root))
        for path in sorted((root / "evaluation").rglob("result.json"))
    ]
    plan = {
        "schema_version": 3,
        "suite": args.study,
        "phase": "held_out_test",
        "seeds": args.seeds,
        "optimizers": list(STANDARD_OPTIMIZERS),
        "tuning_manifest": str(manifest_path.relative_to(root)),
        "results": results,
    }
    write_json(root / "evaluation" / "comparison_manifest.json", plan)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.report",
            str(root / "evaluation"),
            "--output-dir",
            str(root / "evaluation" / "report"),
            "--paper",
        ],
        check=True,
    )


def main(argv=None) -> None:
    """Execute one complete, validation-locked benchmark study."""
    args = parse_args(argv)
    if args.parallelism < 1:
        raise ValueError("--parallelism must be positive")
    if len(set(args.seeds)) < 2:
        raise ValueError("held-out evaluation requires at least two distinct seeds")
    root = Path(args.output_dir)
    spec = STUDIES[args.study]
    manifest_path = root / "tuning_manifest.json"
    if args.phase in ("tune", "all"):
        manifest_path = tune(args, root, spec)
    if args.phase in ("evaluate", "all"):
        if not manifest_path.is_file():
            raise FileNotFoundError("run --phase tune before --phase evaluate")
        evaluate(args, root, spec, manifest_path)


if __name__ == "__main__":
    main()
