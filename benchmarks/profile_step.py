"""Profile reference optimizer-step components and emit a Chrome trace."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks.manifest import reproducibility_metadata, write_json
from neat_optim import ArrayState, NEATConfig
from neat_optim.engine.common import conflict_ratio, safe_projection
from neat_optim.engine.reference import neat_step_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameter-counts", type=int, nargs="+", default=[1_000, 100_000, 1_000_000]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="benchmark-runs/profile-step")
    return parser.parse_args(argv)


def _time(callable_, repeats: int) -> float:
    started = time.perf_counter_ns()
    for _ in range(repeats):
        callable_()
    return (time.perf_counter_ns() - started) / repeats / 1e6


def main(argv=None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows = []
    trace = []
    timestamp = 0.0
    for size in args.parameter_counts:
        gradient = rng.normal(size=size).astype(np.float32)
        opponent = rng.normal(size=size).astype(np.float32)
        parameter = np.zeros(size, dtype=np.float32)
        state = ArrayState.zeros_like(parameter)
        state.momentum[:] = opponent
        config = NEATConfig(native="never")
        correction = safe_projection(gradient, opponent, 1e-8)
        momentum = opponent.copy()
        components = {
            "gradient_read": partial(np.copy, gradient),
            "opponent_proxy": partial(np.copy, opponent),
            "cosine_conflict": partial(conflict_ratio, gradient, opponent, 1e-8),
            "projection": partial(safe_projection, gradient, opponent, 1e-8),
            "momentum_update": partial(
                np.add,
                np.float32(0.9) * momentum,
                np.float32(0.1) * (gradient + correction),
            ),
            "weight_apply": partial(
                np.subtract,
                parameter,
                np.float32(config.learning_rate) * momentum,
            ),
            "complete_step": partial(
                neat_step_reference, parameter, gradient, state, config
            ),
        }
        for name, operation in components.items():
            milliseconds = _time(operation, args.repeats)
            rows.append(
                {
                    "parameter_count": size,
                    "component": name,
                    "milliseconds": milliseconds,
                }
            )
            trace.append(
                {
                    "name": name,
                    "cat": str(size),
                    "ph": "X",
                    "ts": timestamp * 1000,
                    "dur": milliseconds * 1000,
                    "pid": 1,
                    "tid": size,
                }
            )
            timestamp += milliseconds
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "profile.json",
        {
            "config": vars(args),
            "rows": rows,
            "reproducibility": reproducibility_metadata(seed=args.seed),
        },
    )
    (output / "trace.json").write_text(
        json.dumps({"traceEvents": trace}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
