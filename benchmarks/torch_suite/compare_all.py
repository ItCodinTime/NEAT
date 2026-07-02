"""Run NEAT vs AdamW head-to-head across all benchmark categories.

Usage
-----
    # Quick dataset-free smoke-test (all baseline families, five seeds)
    python -m benchmarks.torch_suite.compare_all --quick

    # Single category
    python -m benchmarks.torch_suite.compare_all --category vision

    # Full suite (long — use a GPU node)
    python -m benchmarks.torch_suite.compare_all --output-dir benchmark-runs/full-suite

Each category launches one subprocess per (optimizer × seed) combination and
writes a self-contained result.json per run. After all runs complete a summary
CSV and convergence PNG are generated via the report module.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.manifest import reproducibility_metadata, write_json

OPTIMIZERS = ("neat", "adamw", "sgd", "lion")
TASK_GRADIENT_OPTIMIZERS = ("neat", "pcgrad", "cagrad", "sgd")


def _run(cmd: list[str]) -> None:
    """Echo and execute one fail-fast benchmark subprocess."""
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def _python() -> str:
    """Use the active interpreter so subprocesses share this environment."""
    return sys.executable


def run_vision(
    *,
    output_dir: str,
    dataset: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seeds: list[int],
    target_accuracy: float | None,
    device: str,
    amp: bool,
    data_dir: str,
) -> None:
    """Run and summarize all requested seeds for one vision task."""
    subdir = f"{output_dir}/{dataset}-{model}"
    for seed in seeds:
        for optimizer in OPTIMIZERS:
            cmd = [
                _python(),
                "-m",
                "benchmarks.torch_suite.vision",
                "--dataset",
                dataset,
                "--model",
                model,
                "--optimizer",
                optimizer,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--weight-decay",
                str(weight_decay),
                "--seed",
                str(seed),
                "--output-dir",
                subdir,
                "--device",
                device,
                "--data-dir",
                data_dir,
            ]
            if target_accuracy is not None:
                cmd += ["--target-accuracy", str(target_accuracy)]
            if amp:
                cmd.append("--amp")
            _run(cmd)
    _report(subdir, seeds)


def run_language_model(
    *,
    output_dir: str,
    task: str,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lora: bool,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seeds: list[int],
    device: str,
    amp: bool,
) -> None:
    """Run matched LLM fine-tuning trials for both optimizers."""
    subdir = f"{output_dir}/lm-{task}"
    for seed in seeds:
        for optimizer in OPTIMIZERS:
            cmd = [
                _python(),
                "-m",
                "benchmarks.torch_suite.language_model",
                "--task",
                task,
                "--model",
                model,
                "--optimizer",
                optimizer,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--seed",
                str(seed),
                "--output-dir",
                subdir,
                "--device",
                device,
            ]
            if lora:
                cmd.append("--lora")
            if max_train_samples is not None:
                cmd += ["--max-train-samples", str(max_train_samples)]
            if max_eval_samples is not None:
                cmd += ["--max-eval-samples", str(max_eval_samples)]
            if amp:
                cmd.append("--amp")
            _run(cmd)
    _report(subdir, seeds)


def run_reinforcement_learning(
    *,
    output_dir: str,
    env: str,
    timesteps: int,
    seeds: list[int],
    device: str,
    target_reward: float | None,
) -> None:
    """Run matched SAC trials and report checkpoint reward curves."""
    subdir = f"{output_dir}/rl-{env}"
    for seed in seeds:
        for optimizer in OPTIMIZERS:
            cmd = [
                _python(),
                "-m",
                "benchmarks.torch_suite.reinforcement_learning",
                "--env",
                env,
                "--optimizer",
                optimizer,
                "--timesteps",
                str(timesteps),
                "--seed",
                str(seed),
                "--output-dir",
                subdir,
                "--device",
                device,
            ]
            if target_reward is not None:
                cmd += ["--target-reward", str(target_reward)]
            _run(cmd)
    _report(subdir, seeds)


def run_diffusion(
    *,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seeds: list[int],
    device: str,
    amp: bool,
    data_dir: str,
    target_loss: float | None,
) -> None:
    """Run matched DDPM trials and report validation-loss curves."""
    subdir = f"{output_dir}/diffusion-mnist"
    for seed in seeds:
        for optimizer in OPTIMIZERS:
            cmd = [
                _python(),
                "-m",
                "benchmarks.torch_suite.diffusion",
                "--optimizer",
                optimizer,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(lr),
                "--seed",
                str(seed),
                "--output-dir",
                subdir,
                "--device",
                device,
                "--data-dir",
                data_dir,
            ]
            if target_loss is not None:
                cmd += ["--target-loss", str(target_loss)]
            if amp:
                cmd.append("--amp")
            _run(cmd)
    _report(subdir, seeds)


def run_large_batch(
    *,
    output_dir: str,
    dataset: str,
    model: str,
    batch_sizes: list[int],
    base_lr: float,
    epochs: int,
    seeds: list[int],
    device: str,
    amp: bool,
    data_dir: str,
) -> None:
    """Delegate the batch-size matrix to the dedicated sweep runner."""
    subdir = f"{output_dir}/large-batch-{dataset}-{model}"
    for seed in seeds:
        cmd = [
            _python(),
            "-m",
            "benchmarks.torch_suite.large_batch",
            "--dataset",
            dataset,
            "--model",
            model,
            "--batch-sizes",
            *[str(b) for b in batch_sizes],
            "--base-lr",
            str(base_lr),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--output-dir",
            subdir,
            "--device",
            device,
            "--data-dir",
            data_dir,
        ]
        if amp:
            cmd.append("--amp")
        _run(cmd)
    _report(subdir, seeds)


def run_multitask_synthetic(*, output_dir: str, seeds: list[int], steps: int) -> None:
    """Run baselines requiring explicit task gradients on an applicable task."""
    subdir = f"{output_dir}/multitask-synthetic"
    for seed in seeds:
        for optimizer in TASK_GRADIENT_OPTIMIZERS:
            _run(
                [
                    _python(),
                    "-m",
                    "benchmarks.torch_suite.multitask_synthetic",
                    "--optimizer",
                    optimizer,
                    "--steps",
                    str(steps),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    subdir,
                ]
            )
    write_json(
        Path(subdir) / "comparison_manifest.json",
        {
            "schema_version": 3,
            "suite": "multitask_synthetic",
            "seeds": seeds,
            "optimizers": list(TASK_GRADIENT_OPTIMIZERS),
            "results": [
                str(path.relative_to(subdir))
                for path in Path(subdir).rglob("result.json")
            ],
        },
    )
    _report(subdir, seeds, TASK_GRADIENT_OPTIMIZERS)


def run_optimizer_smoke(*, output_dir: str, seeds: list[int], steps: int) -> None:
    """Run every standard optimizer without external datasets."""
    subdir = f"{output_dir}/optimizer-smoke"
    for seed in seeds:
        for optimizer in OPTIMIZERS:
            _run(
                [
                    _python(),
                    "-m",
                    "benchmarks.torch_suite.optimizer_smoke",
                    "--optimizer",
                    optimizer,
                    "--steps",
                    str(steps),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    subdir,
                ]
            )
    _report(subdir, seeds)


def _report(
    subdir: str,
    seeds: list[int],
    optimizers: tuple[str, ...] = OPTIMIZERS,
) -> None:
    """Generate summary artifacts while preserving raw runs on failure."""
    plan = Path(subdir) / "comparison_manifest.json"
    write_json(
        plan,
        {
            "schema_version": 3,
            "suite": Path(subdir).name,
            "seeds": seeds,
            "optimizers": list(optimizers),
            "results": [
                str(path.relative_to(subdir))
                for path in Path(subdir).rglob("result.json")
            ],
        },
    )
    report_dir = str(Path(subdir) / "report")
    try:
        subprocess.run(
            [
                _python(),
                "-m",
                "benchmarks.report",
                subdir,
                "--output-dir",
                report_dir,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"[compare_all] report failed for {subdir} — raw data preserved.")


def parse_args(argv=None):
    """Parse suite selection and shared execution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        choices=(
            "vision",
            "vision-vit",
            "language-model",
            "rl",
            "diffusion",
            "large-batch",
            "multitask",
            "optimizer-smoke",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Minimal epochs for smoke-testing"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 29])
    parser.add_argument(
        "--allow-single-seed",
        action="store_true",
        help="Permit a preliminary run that cannot be used for paper statistics",
    )
    parser.add_argument("--output-dir", default="benchmark-runs/compare-all")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """Dispatch selected benchmark categories with credible default budgets."""
    args = parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must not contain duplicates")
    if len(args.seeds) == 1 and not args.allow_single_seed:
        raise SystemExit("single-seed runs require --allow-single-seed")
    os.environ["NEAT_BENCHMARK_ORCHESTRATED"] = "1"
    if len(args.seeds) == 1:
        os.environ["NEAT_BENCHMARK_PRELIMINARY"] = "1"
    started = time.time()

    if args.category in ("optimizer-smoke", "all"):
        run_optimizer_smoke(
            output_dir=args.output_dir,
            seeds=args.seeds,
            steps=10 if args.quick else 100,
        )

    run_heavyweight = args.category != "all" or not args.quick

    if args.category in ("multitask", "all"):
        run_multitask_synthetic(
            output_dir=args.output_dir,
            seeds=args.seeds,
            steps=10 if args.quick else 100,
        )

    # -- Vision (CIFAR-10, ResNet-18) -----------------------------------------
    if args.category == "vision" or (args.category == "all" and run_heavyweight):
        run_vision(
            output_dir=args.output_dir,
            dataset="cifar10",
            model="resnet18",
            epochs=3 if args.quick else 100,
            batch_size=128,
            lr=1e-3,
            weight_decay=5e-4,
            seeds=args.seeds,
            target_accuracy=0.92 if not args.quick else None,
            device=args.device,
            amp=args.amp,
            data_dir=args.data_dir,
        )
        # CIFAR-100 with ResNet-34 (full only)
        if not args.quick:
            run_vision(
                output_dir=args.output_dir,
                dataset="cifar100",
                model="resnet34",
                epochs=200,
                batch_size=128,
                lr=1e-3,
                weight_decay=5e-4,
                seeds=args.seeds,
                target_accuracy=0.70,
                device=args.device,
                amp=args.amp,
                data_dir=args.data_dir,
            )

    # -- Vision Transformer (ViT-small, CIFAR-100) ----------------------------
    if args.category == "vision-vit" or (args.category == "all" and run_heavyweight):
        run_vision(
            output_dir=args.output_dir,
            dataset="cifar100",
            model="vit_small",
            epochs=1 if args.quick else 200,
            batch_size=128,
            lr=5e-4,
            weight_decay=0.05,
            seeds=args.seeds,
            target_accuracy=0.70 if not args.quick else None,
            device=args.device,
            amp=args.amp,
            data_dir=args.data_dir,
        )

    # -- Language Model (TinyLlama SST-2) -------------------------------------
    if args.category == "language-model" or (
        args.category == "all" and run_heavyweight
    ):
        run_language_model(
            output_dir=args.output_dir,
            task="sst2",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            epochs=1 if args.quick else 3,
            batch_size=16,
            lr=2e-5,
            lora=True,
            max_train_samples=200 if args.quick else None,
            max_eval_samples=100 if args.quick else None,
            seeds=args.seeds,
            device=args.device,
            amp=args.amp,
        )

    # -- Reinforcement Learning (HalfCheetah) ---------------------------------
    if args.category == "rl" or (args.category == "all" and run_heavyweight):
        run_reinforcement_learning(
            output_dir=args.output_dir,
            env="HalfCheetah-v5",
            timesteps=5_000 if args.quick else 1_000_000,
            seeds=args.seeds,
            device=args.device,
            target_reward=None if args.quick else 8000,
        )

    # -- Diffusion (MNIST) ----------------------------------------------------
    if args.category == "diffusion" or (args.category == "all" and run_heavyweight):
        run_diffusion(
            output_dir=args.output_dir,
            epochs=1 if args.quick else 50,
            batch_size=128,
            lr=1e-4,
            seeds=args.seeds,
            device=args.device,
            amp=args.amp,
            data_dir=args.data_dir,
            target_loss=None if args.quick else 0.015,
        )

    # -- Large-Batch Stability sweep ------------------------------------------
    if args.category == "large-batch" or (args.category == "all" and run_heavyweight):
        run_large_batch(
            output_dir=args.output_dir,
            dataset="cifar10",
            model="resnet18",
            batch_sizes=[128, 256] if args.quick else [128, 512, 2048],
            base_lr=1e-3,
            epochs=1 if args.quick else 100,
            seeds=args.seeds,
            device=args.device,
            amp=args.amp,
            data_dir=args.data_dir,
        )

    wall_time = time.time() - started
    merged = {
        "schema_version": 3,
        "suite": "compare_all",
        "category": args.category,
        "quick": args.quick,
        "optimizers": list(OPTIMIZERS),
        "task_gradient_optimizers": list(TASK_GRADIENT_OPTIMIZERS),
        "seeds": args.seeds,
        "preliminary": len(args.seeds) == 1,
        "wall_time_seconds": wall_time,
        "reproducibility": reproducibility_metadata(seed=args.seeds[0]),
        "results": [
            str(path.relative_to(args.output_dir))
            for path in Path(args.output_dir).rglob("result.json")
        ],
    }
    write_json(Path(args.output_dir) / "comparison_manifest.json", merged)
    print(f"\n[compare_all] Finished in {wall_time / 60:.1f} min.")
    print(f"[compare_all] Results under: {args.output_dir}")
    print(
        f"[compare_all] Aggregate report: "
        f"python -m benchmarks.torch_suite.report {args.output_dir}"
    )


if __name__ == "__main__":
    main()
