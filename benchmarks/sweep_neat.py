from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.tasks.keras_mlp import run_neat_sweep


def main() -> None:
    print(json.dumps(run_neat_sweep(), indent=2))


if __name__ == "__main__":
    main()
