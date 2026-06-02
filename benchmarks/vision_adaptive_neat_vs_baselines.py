"""Run a short multi-seed vision benchmark for adaptive NEAT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.tasks.keras_vision import (
    VisionBenchmarkConfig,
    run_benchmark,
)


def main() -> None:
    result = run_benchmark(
        VisionBenchmarkConfig(
            datasets=("mnist", "fashion_mnist"),
            seeds=(7, 11, 19),
            epochs=2,
            batch_size=128,
            validation_size=5000,
        )
    )
    output = Path(
        f"benchmarks/results/vision_adaptive_neat_standard_{result['date']}.json"
    )
    output.write_text(json.dumps(result, indent=2))
    print(output)
    for row in result["summary"]:
        print(
            row["dataset"],
            row["optimizer"],
            row["mean_test_accuracy"],
            row["mean_test_loss"],
            row["mean_seconds"],
        )


if __name__ == "__main__":
    main()
