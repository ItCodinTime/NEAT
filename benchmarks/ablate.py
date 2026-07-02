"""Run one-at-a-time NEAT ablations on the deterministic conflict testbed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from benchmarks.manifest import reproducibility_metadata, write_json
from benchmarks.synthetic_runner import run_synthetic
from neat_optim import NEATConfig

ABLATION_KNOBS = {
    "opponent_source",
    "correction_warmup_steps",
    "conflict_threshold",
    "adaptive_alpha",
    "adaptive_preconditioning",
    "update_mode",
    "gradient_centralization",
    "nesterov",
    "lookahead_k",
    "conflict_granularity",
    "trust_region_ratio",
    "opponent_ensemble",
    "opponent_ensemble_momentum_weight",
    "opponent_ensemble_schedule_steps",
    "correction_preconditioning_order",
    "deterministic",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--sweep",
        required=True,
        help="JSON object mapping each knob to the values evaluated",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 29])
    parser.add_argument("--allow-single-seed", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dimensionality", type=int, default=32)
    parser.add_argument("--conflict-intensity", type=float, required=True)
    parser.add_argument("--noise-level", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output-dir", default="benchmark-runs/ablations")
    return parser.parse_args(argv)


def one_at_a_time(
    base: dict[str, Any], sweep: dict[str, list[Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """Return a base case and configurations changing exactly one key."""
    unknown = set(sweep) - ABLATION_KNOBS
    if unknown:
        raise ValueError(f"unsupported ablation knobs: {sorted(unknown)}")
    configurations = [("base", base.copy())]
    for knob, values in sweep.items():
        for value in values:
            candidate = base.copy()
            candidate[knob] = value
            configurations.append((f"{knob}={value}", candidate))
    return configurations


def main(argv=None) -> None:
    args = parse_args(argv)
    if len(args.seeds) == 1 and not args.allow_single_seed:
        raise SystemExit("single-seed runs require --allow-single-seed")
    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    sweep = json.loads(args.sweep)
    output = Path(args.output_dir)
    paths = []
    for label, values in one_at_a_time(base, sweep):
        for seed in args.seeds:
            started = time.perf_counter()
            config = NEATConfig(**values)
            result = run_synthetic(
                config=config,
                seed=seed,
                steps=args.steps,
                dimensionality=args.dimensionality,
                conflict_intensity=args.conflict_intensity,
                noise_level=args.noise_level,
                batch_size=args.batch_size,
            )
            manifest = {
                "schema_version": 3,
                "config": {
                    **vars(args),
                    "optimizer": label,
                    "seed": seed,
                    "task": "synthetic_ablation",
                },
                "history": result["history"],
                "wall_time_seconds": time.perf_counter() - started,
                "preliminary": len(args.seeds) == 1,
                "reproducibility": reproducibility_metadata(seed=seed),
                "synthetic": result,
            }
            path = output / label.replace("/", "_") / f"seed{seed}" / "result.json"
            write_json(path, manifest)
            paths.append(str(path.relative_to(output)))
    write_json(
        output / "comparison_manifest.json",
        {
            "schema_version": 3,
            "suite": "ablation",
            "seeds": args.seeds,
            "optimizers": [label for label, _ in one_at_a_time(base, sweep)],
            "results": paths,
        },
    )


if __name__ == "__main__":
    main()
