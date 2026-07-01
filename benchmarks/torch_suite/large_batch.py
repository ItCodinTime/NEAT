"""Launch a controlled CIFAR batch-size/optimizer stability sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args(argv=None):
    """Parse the batch sizes and fixed vision configuration to sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(128, 512, 2048))
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=("neat", "adamw"),
        default=("neat", "adamw"),
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100"), default="cifar100"
    )
    parser.add_argument(
        "--model",
        choices=("resnet18", "resnet34", "vit_small", "deit_small"),
        default="vit_small",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="benchmark-runs/large-batch")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """Launch matched optimizer trials for every requested batch size."""
    args = parse_args(argv)
    for batch_size in args.batch_sizes:
        # Linear scaling makes the batch-size comparison explicit and reproducible.
        learning_rate = args.base_lr * batch_size / min(args.batch_sizes)
        for optimizer in args.optimizers:
            command = [
                sys.executable,
                "-m",
                "benchmarks.torch_suite.vision",
                "--dataset",
                args.dataset,
                "--model",
                args.model,
                "--optimizer",
                optimizer,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(batch_size),
                "--lr",
                str(learning_rate),
                "--seed",
                str(args.seed),
                "--output-dir",
                args.output_dir,
                "--data-dir",
                args.data_dir,
                "--device",
                args.device,
            ]
            if args.amp:
                command.append("--amp")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
