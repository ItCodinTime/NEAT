"""Compare NumPy reference and optional fused-native optimizer step timings."""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks.manifest import reproducibility_metadata, write_json
from neat_optim import ArrayState, NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.engine.reference import neat_step_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-counts", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="benchmark-runs/native/result.json")
    return parser.parse_args(argv)


def _measure(operation, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return float(np.mean(samples))


def _reference_once(parameter: np.ndarray, gradient: np.ndarray) -> None:
    neat_step_reference(
        parameter.copy(),
        gradient,
        ArrayState.zeros_like(parameter),
        NEATConfig(native="never"),
    )


def _native_once(parameter: np.ndarray, gradient: np.ndarray) -> None:
    neat_step(
        parameter.copy(),
        gradient,
        ArrayState.zeros_like(parameter),
        NEATConfig(native="force", native_fused=True),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows = []
    for size in args.parameter_counts:
        parameter = rng.normal(size=size).astype(np.float32)
        gradient = rng.normal(size=size).astype(np.float32)

        reference_step = partial(_reference_once, parameter, gradient)
        native_step = partial(_native_once, parameter, gradient)

        rows.append(
            {
                "parameter_count": size,
                "reference_milliseconds": _measure(reference_step, args.repeats),
                "fused_native_milliseconds": _measure(native_step, args.repeats),
            }
        )
    write_json(
        Path(args.output),
        {
            "config": vars(args),
            "rows": rows,
            "reproducibility": reproducibility_metadata(seed=args.seed),
        },
    )


if __name__ == "__main__":
    main()
