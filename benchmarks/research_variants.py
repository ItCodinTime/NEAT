"""Run research-oriented NEAT benchmark variants against standard baselines."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tasks.keras_mlp import (
    BenchmarkConfig,
    OptimizerSpec,
    run_optimizer_suite,
)


def build_research_specs() -> tuple[OptimizerSpec, ...]:
    """Return the targeted optimizer variants used for NEAT research runs."""
    return (
        OptimizerSpec("sgd_momentum", "sgd_momentum", {"learning_rate": 1e-2}),
        OptimizerSpec("adam", "adam", {"learning_rate": 1e-3}),
        OptimizerSpec(
            "neat_best_prev",
            "neat",
            {
                "learning_rate": 0.06,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "opponent_source": "momentum",
                "correction_warmup_steps": 0,
                "conflict_threshold": 0.0,
            },
        ),
        OptimizerSpec(
            "neat_blended_adaptive",
            "neat",
            {
                "learning_rate": 0.06,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "opponent_source": "blended",
                "opponent_blend": 0.25,
                "adaptive_correction": True,
                "adaptive_correction_decay": 0.9,
                "adaptive_correction_min_scale": 1.0,
                "adaptive_correction_max_scale": 2.5,
                "correction_warmup_steps": 0,
                "conflict_threshold": 0.0,
            },
        ),
        OptimizerSpec(
            "neat_prevgrad_adaptive",
            "neat",
            {
                "learning_rate": 0.06,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "opponent_source": "previous_gradient",
                "adaptive_correction": True,
                "adaptive_correction_decay": 0.9,
                "adaptive_correction_min_scale": 1.0,
                "adaptive_correction_max_scale": 2.5,
                "correction_warmup_steps": 0,
                "conflict_threshold": 0.0,
            },
        ),
        OptimizerSpec(
            "player_neat_adaptive",
            "player_neat",
            {
                "learning_rate": 0.06,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "adaptive_correction": True,
                "adaptive_correction_decay": 0.9,
                "adaptive_correction_min_scale": 1.0,
                "adaptive_correction_max_scale": 2.5,
                "opponent_mode": "mean_excluding_self",
                "player_reduction": "mean",
            },
        ),
    )


def main() -> None:
    result = run_optimizer_suite(
        build_research_specs(),
        config=BenchmarkConfig(
            seeds=(7, 11, 19),
            epochs=20,
            batch_size=64,
            player_batch_size=32,
        ),
    )
    output = Path("benchmarks/results/neat_research_variants_2026-04-05.json")
    output.write_text(json.dumps(result, indent=2))
    print(output)
    for row in result["summary"]:
        print(
            row["optimizer"],
            row["mean_test_accuracy"],
            row["mean_test_loss"],
            row["mean_seconds"],
            row["mean_conflict_ratio"],
            row["mean_correction_ratio"],
        )


if __name__ == "__main__":
    main()
