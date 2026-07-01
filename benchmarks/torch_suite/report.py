"""Aggregate result manifests and render convergence/stability curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args(argv=None):
    """Parse input run trees and the report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="result.json files or directories")
    parser.add_argument("--output-dir", default="benchmark-runs/report")
    return parser.parse_args(argv)


def discover(inputs: list[str]) -> list[Path]:
    """Expand files and directories into unique result manifests."""
    paths = []
    for value in inputs:
        candidate = Path(value)
        paths.extend(
            candidate.rglob("result.json") if candidate.is_dir() else [candidate]
        )
    return sorted(set(paths))


def main(argv=None):
    """Write a flat CSV summary and optional convergence figure."""
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in discover(args.inputs)
    ]
    if not results:
        raise SystemExit("No result.json manifests found.")
    fields = [
        "run", "task", "optimizer", "seed", "final_metric", "best_metric",
        "epochs_to_target", "final_loss", "mean_loss_variance", "wall_time_seconds",
    ]
    rows = []
    for path, result in results:
        config, history = result["config"], result["history"]
        higher = result.get("metric") not in {"perplexity", "validation_noise_mse"}
        metrics = [row["eval_metric"] for row in history]
        rows.append({
            "run": str(path.parent),
            "task": config.get(
                "task", config.get("dataset", config.get("env", "unknown"))
            ),
            "optimizer": config["optimizer"],
            "seed": config["seed"],
            "final_metric": metrics[-1],
            "best_metric": (max(metrics) if higher else min(metrics)),
            "epochs_to_target": result.get("epochs_to_target"),
            "final_loss": history[-1]["train_loss"],
            "mean_loss_variance": sum(
                row["loss_variance"] for row in history
            )
            / len(history),
            "wall_time_seconds": result["wall_time_seconds"],
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Wrote {output / 'summary.csv'} (install matplotlib for plots)")
        return
    figure, (metric_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 4))
    for path, result in results:
        history, config = result["history"], result["config"]
        label = f"{config['optimizer']}/seed{config['seed']}"
        epochs = [row["epoch"] for row in history]
        metric_axis.plot(epochs, [row["eval_metric"] for row in history], label=label)
        loss_axis.plot(epochs, [row["train_loss"] for row in history], label=label)
    metric_axis.set(
        title="Evaluation convergence", xlabel="epoch/checkpoint", ylabel="metric"
    )
    loss_axis.set(
        title="Training convergence", xlabel="epoch/checkpoint", ylabel="loss"
    )
    metric_axis.legend()
    loss_axis.legend()
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=160)
    print(f"Wrote {output / 'summary.csv'} and {output / 'convergence.png'}")


if __name__ == "__main__":
    main()
