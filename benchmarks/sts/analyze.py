"""Aggregate STS manifests into statistical tables and mechanism plots."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.manifest import write_json
from benchmarks.stats import paired_comparison, summarize
from benchmarks.sts import DATASETS, METHODS, PAPER_SEEDS

METRICS = ("task_1_accuracy", "task_2_accuracy", "average_accuracy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="benchmark-runs/sts")
    parser.add_argument("--output-dir", default="paper_assets/sts")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    return parser.parse_args(argv)


def load_results(runs_dir: str) -> list[dict]:
    rows = []
    for path in sorted(Path(runs_dir).glob("runs/**/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        rows.append(payload)
    return rows


def core_results(rows: list[dict], seeds: list[int]) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["config"]["setting"] == "multi"
        and row["config"]["neat_variant"] == "base"
        and row["config"]["seed"] in seeds
        and row["test"] is not None
        and not row["preliminary"]
    ]
    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in seeds
    }
    actual = {
        (row["config"]["dataset"], row["config"]["method"], row["config"]["seed"])
        for row in selected
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"incomplete Tier 1 matrix; missing={missing}, extra={extra}")
    return selected


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_core(rows: list[dict]):
    summaries = []
    tests = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["config"]["dataset"] == dataset]
        for method in METHODS:
            method_rows = [
                row for row in dataset_rows if row["config"]["method"] == method
            ]
            for metric in METRICS:
                summary = summarize(row["test"][metric] for row in method_rows)
                summaries.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        **asdict(summary),
                    }
                )
        for baseline in (method for method in METHODS if method != "neat"):
            for metric in METRICS:
                neat = {
                    row["config"]["seed"]: row["test"][metric]
                    for row in dataset_rows
                    if row["config"]["method"] == "neat"
                }
                other = {
                    row["config"]["seed"]: row["test"][metric]
                    for row in dataset_rows
                    if row["config"]["method"] == baseline
                }
                tests.append(
                    {
                        "dataset": dataset,
                        "comparison": f"neat_minus_{baseline}",
                        "metric": metric,
                        "mean_difference": float(
                            np.mean([neat[seed] - other[seed] for seed in neat])
                        ),
                        **paired_comparison(neat, other),
                    }
                )
    return summaries, tests


def cost_rows(core: list[dict]) -> list[dict]:
    output = []
    for dataset in DATASETS:
        by_method = {
            method: [
                row
                for row in core
                if row["config"]["dataset"] == dataset
                and row["config"]["method"] == method
            ]
            for method in METHODS
        }
        adam_time = np.mean(
            [row["mean_step_time_seconds"] for row in by_method["adam"]]
        )
        adam_memory = np.mean([row["peak_memory_mb"] for row in by_method["adam"]])
        for method, rows in by_method.items():
            step_time = float(np.mean([row["mean_step_time_seconds"] for row in rows]))
            memory = float(np.mean([row["peak_memory_mb"] for row in rows]))
            output.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_step_time_seconds": step_time,
                    "time_overhead_vs_adam_percent": 100 * (step_time / adam_time - 1),
                    "mean_peak_memory_mb": memory,
                    "memory_overhead_vs_adam_percent": 100 * (memory / adam_memory - 1),
                }
            )
    return output


def plot_mechanism(rows: list[dict], core: list[dict], output_dir: Path) -> dict:
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    output_dir.mkdir(parents=True, exist_ok=True)
    neat_rows = [
        row
        for row in rows
        if row["config"]["method"] == "neat"
        and row["config"]["neat_variant"] == "base"
        and row["config"]["setting"] in {"multi", "single_task_1", "single_task_2"}
        and not row["preliminary"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"multi": "#c43c39", "single": "#2878b5"}
    for axis, dataset in zip(axes, DATASETS, strict=True):
        selected = [row for row in neat_rows if row["config"]["dataset"] == dataset]
        for label, settings in (
            ("multi-task", {"multi"}),
            ("single-task", {"single_task_1", "single_task_2"}),
        ):
            relevant = [row for row in selected if row["config"]["setting"] in settings]
            if not relevant:
                continue
            epochs = [row["epoch"] for row in relevant[0]["history"]]
            values = np.asarray(
                [
                    [point["correction_ratio"] for point in row["history"]]
                    for row in relevant
                ]
            )
            mean, std = values.mean(0), values.std(0, ddof=1)
            key = "multi" if label == "multi-task" else "single"
            axis.plot(epochs, mean, label=label, color=colors[key])
            axis.fill_between(
                epochs, mean - std, mean + std, color=colors[key], alpha=0.18
            )
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("NEAT correction ratio")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "correction_ratio_single_vs_multi.png", dpi=200)
    plt.close(fig)

    points = []
    for neat in (row for row in core if row["config"]["method"] == "neat"):
        config = neat["config"]
        adam = next(
            row
            for row in core
            if row["config"]["method"] == "adam"
            and row["config"]["dataset"] == config["dataset"]
            and row["config"]["seed"] == config["seed"]
        )
        points.append(
            {
                "dataset": config["dataset"],
                "seed": config["seed"],
                "correction_ratio": float(
                    np.mean([point["correction_ratio"] for point in neat["history"]])
                ),
                "performance_gain": neat["test"]["average_accuracy"]
                - adam["test"]["average_accuracy"],
            }
        )
    x = np.asarray([point["correction_ratio"] for point in points])
    y = np.asarray([point["performance_gain"] for point in points])
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    for dataset, marker in zip(DATASETS, ("o", "s"), strict=True):
        subset = [point for point in points if point["dataset"] == dataset]
        axis.scatter(
            [point["correction_ratio"] for point in subset],
            [point["performance_gain"] for point in subset],
            label=dataset.replace("_", " "),
            marker=marker,
        )
    if np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line = np.linspace(x.min(), x.max(), 100)
        axis.plot(line, slope * line + intercept, color="#333333", linewidth=1)
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Mean NEAT correction ratio")
    axis.set_ylabel("Accuracy gain over paired Adam run")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "correction_ratio_vs_performance_gain.png", dpi=200)
    plt.close(fig)
    return {
        "points": points,
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
        "interpretation_limit": (
            "Association and dose-response evidence do not alone establish causation."
        ),
    }


def markdown_table(summaries: list[dict], tests: list[dict]) -> str:
    lines = [
        "# Regeneron STS Tier 1 Results",
        "",
        "All values are test accuracy mean +/- sample standard deviation "
        "across five fixed seeds.",
        "",
        "| Dataset | Method | Task 1 | Task 2 | Average |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            values = {
                row["metric"]: row
                for row in summaries
                if row["dataset"] == dataset and row["method"] == method
            }
            formatted = [
                f"{values[metric]['mean']:.4f} +/- {values[metric]['std']:.4f}"
                for metric in METRICS
            ]
            lines.append(f"| {dataset} | {method} | " + " | ".join(formatted) + " |")
    lines.extend(["", "## Paired NEAT comparisons", ""])
    for row in tests:
        if row["metric"] == "average_accuracy":
            lines.append(
                f"- {row['dataset']} {row['comparison']}: mean difference "
                f"{row['mean_difference']:.6f}, paired t-test "
                f"p={row['paired_t_pvalue']:.6g}, "
                f"paired Cohen's d={row['cohens_d_paired']:.4f}."
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    rows = load_results(args.runs_dir)
    core = core_results(rows, args.seeds)
    summaries, tests = summarize_core(core)
    costs = cost_rows(core)
    mechanism = plot_mechanism(rows, core, output_dir / "plots")
    _write_csv(output_dir / "tier1_summary.csv", summaries)
    _write_csv(output_dir / "paired_tests.csv", tests)
    _write_csv(output_dir / "cost_summary.csv", costs)
    write_json(output_dir / "mechanism_statistics.json", mechanism)
    write_json(
        output_dir / "full_precision_results.json",
        {"summaries": summaries, "paired_tests": tests, "costs": costs},
    )
    (output_dir / "tier1_results.md").write_text(
        markdown_table(summaries, tests), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
