"""Generate complete multi-seed Markdown, LaTeX, JSON, and PDF reports."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.manifest import validate_paper_eligible, write_json
from benchmarks.stats import paired_comparison, require_complete_seeds, summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default="paper_assets")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--paper", action="store_true")
    return parser.parse_args(argv)


def _discover(
    inputs: list[str],
) -> tuple[list[Path], list[dict[str, Any]]]:
    results: list[Path] = []
    plans: list[dict[str, Any]] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            results.extend(path.rglob("result.json"))
            for comparison in path.rglob("comparison_manifest.json"):
                payload = json.loads(comparison.read_text(encoding="utf-8"))
                plans.append(
                    {
                        "directory": comparison.parent.resolve(),
                        "seeds": [int(seed) for seed in payload.get("seeds", [])],
                        "optimizers": list(payload.get("optimizers", [])),
                        "results": list(payload.get("results", [])),
                    }
                )
        else:
            results.append(path)
    return sorted(set(results)), plans


def _plan_for(path: str, plans: list[dict[str, Any]]) -> dict[str, Any] | None:
    resolved = Path(path).resolve()
    candidates = [plan for plan in plans if plan["directory"] in resolved.parents]
    if not candidates:
        return None
    return max(candidates, key=lambda plan: len(plan["directory"].parts))


def _row(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    history = manifest["history"]
    if not history:
        raise ValueError(f"empty history: {path}")
    config = manifest["config"]
    task_name = config.get("task", config.get("dataset", config.get("env", "unknown")))
    task_parts = [str(task_name)]
    if config.get("model"):
        task_parts.append(str(config["model"]))
    if config.get("batch_size") is not None:
        task_parts.append(f"batch{config['batch_size']}")
    return {
        "path": str(path),
        "task": "/".join(task_parts),
        "optimizer": config["optimizer"],
        "seed": int(config["seed"]),
        "final_metric": float(history[-1]["eval_metric"]),
        "wall_time_seconds": float(manifest["wall_time_seconds"]),
        "history": history,
        "tuning_status": config.get("tuning_status", "unspecified"),
    }


def _stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _require_uniform_tuning(task: str, rows: list[dict[str, Any]]) -> None:
    statuses = {row["tuning_status"] for row in rows}
    if len(statuses) != 1:
        raise ValueError(f"{task}: comparison mixes tuning statuses {sorted(statuses)}")


def _render_tables(
    aggregates: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    output: Path,
) -> None:
    markdown = [
        "# Benchmark summary",
        "",
        "| Task | Optimizer | Metric (mean ± std) | 95% CI |",
        "|---|---|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{llrr}",
        r"Task & Optimizer & Mean $\pm$ std & 95\% CI \\",
        r"\hline",
    ]
    for row in aggregates:
        summary = row["final_metric"]
        display = f"{summary['mean']:.4g} ± {summary['std']:.3g}"
        interval = f"[{summary['ci95_low']:.4g}, {summary['ci95_high']:.4g}]"
        markdown.append(
            f"| {row['task']} | {row['optimizer']} | {display} | {interval} |"
        )
        latex.append(
            f"{row['task']} & {row['optimizer']} & ${display}$ & ${interval}$ \\\\"
        )
    latex.append(r"\end{tabular}")
    markdown.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Stars use the paired t-test; Wilcoxon p-values are shown separately.",
            "",
            "| Task | Pair | Paired t p | Wilcoxon p | Paired Cohen's d |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        stars = _stars(row["paired_t_pvalue"])
        markdown.append(
            f"| {row['task']} | {row['left']} vs {row['right']} | "
            f"{row['paired_t_pvalue']:.3g}{stars} | "
            f"{row['wilcoxon_pvalue']:.3g} | {row['cohens_d_paired']:.3g} |"
        )
    (output / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output / "summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def _render_figures(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    from benchmarks.style import PALETTE, apply_style

    apply_style()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    for task, task_rows in sorted(by_task.items()):
        by_optimizer: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_optimizer[row["optimizer"]].append(row)
        stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task).strip("-").lower()
        figure, axis = plt.subplots()
        for index, (optimizer, optimizer_rows) in enumerate(
            sorted(by_optimizer.items())
        ):
            length = min(len(row["history"]) for row in optimizer_rows)
            curves = np.asarray(
                [
                    [point["eval_metric"] for point in row["history"][:length]]
                    for row in optimizer_rows
                ]
            )
            x = np.asarray(
                [point["epoch"] for point in optimizer_rows[0]["history"][:length]]
            )
            mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            color = PALETTE[index % len(PALETTE)]
            axis.plot(x, mean, label=optimizer, color=color)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        axis.set(
            title=task,
            xlabel="Epoch/checkpoint",
            ylabel="Evaluation metric",
        )
        axis.legend()
        figure.savefig(output / f"{stem}-accuracy-curves.pdf")
        plt.close(figure)

        figure, axis = plt.subplots()
        for index, (optimizer, optimizer_rows) in enumerate(
            sorted(by_optimizer.items())
        ):
            axis.scatter(
                [row["wall_time_seconds"] for row in optimizer_rows],
                [row["final_metric"] for row in optimizer_rows],
                label=optimizer,
                color=PALETTE[index % len(PALETTE)],
            )
        axis.set(
            title=task,
            xlabel="Wall time (seconds)",
            ylabel="Final evaluation metric",
        )
        axis.legend()
        figure.savefig(output / f"{stem}-accuracy-vs-wall-time.pdf")
        plt.close(figure)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths, plans = _discover(args.inputs)
    for plan in plans:
        missing = [
            result
            for result in plan["results"]
            if not (plan["directory"] / result).is_file()
        ]
        if missing:
            raise ValueError(
                f"comparison manifest in {plan['directory']} references missing "
                f"results: {missing}"
            )
    if not paths:
        raise SystemExit("No result.json manifests found")
    manifests = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    if args.paper:
        for _, manifest in manifests:
            validate_paper_eligible(manifest)
    rows = [_row(path, manifest) for path, manifest in manifests]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    aggregates = []
    comparisons = []
    for task, task_rows in sorted(grouped.items()):
        _require_uniform_tuning(task, task_rows)
        task_plan = _plan_for(task_rows[0]["path"], plans)
        expected = args.seeds or (task_plan and task_plan["seeds"])
        if not expected:
            raise SystemExit(
                f"Seed plan unavailable for {task}; pass --seeds or include a plan"
            )
        indexed = require_complete_seeds(task_rows, expected)
        planned_optimizers = task_plan and task_plan["optimizers"]
        if planned_optimizers and set(indexed) != set(planned_optimizers):
            raise ValueError(
                f"{task}: optimizer set {sorted(indexed)} does not match plan "
                f"{planned_optimizers}"
            )
        for optimizer, seeded in sorted(indexed.items()):
            aggregates.append(
                {
                    "task": task,
                    "optimizer": optimizer,
                    "final_metric": asdict(
                        summarize(row["final_metric"] for row in seeded.values())
                    ),
                }
            )
        optimizers = sorted(indexed)
        for left_index, left in enumerate(optimizers):
            for right in optimizers[left_index + 1 :]:
                comparisons.append(
                    {
                        "task": task,
                        "left": left,
                        "right": right,
                        **paired_comparison(
                            {
                                seed: row["final_metric"]
                                for seed, row in indexed[left].items()
                            },
                            {
                                seed: row["final_metric"]
                                for seed, row in indexed[right].items()
                            },
                        ),
                    }
                )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "aggregate.json",
        {
            "requested_seeds": args.seeds,
            "aggregates": aggregates,
            "paired_comparisons": comparisons,
        },
    )
    _render_tables(aggregates, comparisons, output)
    _render_figures(rows, output)


if __name__ == "__main__":
    main()
