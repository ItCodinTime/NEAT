"""Run a short GLUE SST-2 benchmark for adaptive NEAT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.tasks.glue_sst2 import GlueSst2BenchmarkConfig, run_benchmark


def main() -> None:
    result = run_benchmark(
        GlueSst2BenchmarkConfig(
            seeds=(7, 11, 19),
            epochs=2,
            batch_size=128,
            max_tokens=20000,
            sequence_length=64,
            embedding_dim=128,
        )
    )
    output = Path(f"benchmarks/results/glue_sst2_adaptive_neat_{result['date']}.json")
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
