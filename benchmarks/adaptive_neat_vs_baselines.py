"""Run the confirmed adaptive-NEAT comparison against Adam and SGD."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.tasks.keras_mlp import (
    BenchmarkConfig,
    OptimizerSpec,
    run_optimizer_suite,
)


def main() -> None:
    specs = (
        OptimizerSpec("sgd_momentum", "sgd_momentum", {"learning_rate": 1e-2}),
        OptimizerSpec("adam", "adam", {"learning_rate": 1e-3}),
        OptimizerSpec(
            "neat_adaptive_best",
            "neat",
            {
                "learning_rate": 0.008,
                "alpha": 0.25,
                "beta": 0.9,
                "opponent_source": "previous_gradient",
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "adaptive_correction": True,
                "adaptive_correction_decay": 0.9,
                "adaptive_correction_min_scale": 1.0,
                "adaptive_correction_max_scale": 2.5,
                "adaptive_preconditioning": True,
                "second_moment_beta": 0.999,
                "bias_correction": True,
                "precondition_nce": True,
                "correction_warmup_steps": 0,
                "conflict_threshold": 0.0,
            },
        ),
    )
    result = run_optimizer_suite(
        specs,
        config=BenchmarkConfig(seeds=(7, 11, 19), epochs=20, batch_size=64),
    )
    output = Path("benchmarks/results/neat_adaptive_confirm_2026-04-05.json")
    output.write_text(json.dumps(result, indent=2))
    print(output)
    for row in result["summary"]:
        print(
            row["optimizer"],
            row["mean_test_accuracy"],
            row["mean_test_loss"],
            row["mean_seconds"],
        )


if __name__ == "__main__":
    main()
