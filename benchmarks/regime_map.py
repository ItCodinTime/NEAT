"""Generate a synthetic batch-size/noise/conflict regime map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmarks.manifest import reproducibility_metadata, write_json
from benchmarks.synthetic_runner import run_synthetic
from neat_optim import NEATConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--noise-levels", type=float, nargs="+", required=True)
    parser.add_argument("--conflict-intensities", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19, 23, 29])
    parser.add_argument("--allow-single-seed", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dimensionality", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", default="benchmark-runs/regime-map")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if len(args.seeds) == 1 and not args.allow_single_seed:
        raise SystemExit("single-seed runs require --allow-single-seed")
    rows = []
    for batch_size in args.batch_sizes:
        for noise in args.noise_levels:
            for conflict in args.conflict_intensities:
                for seed in args.seeds:
                    shared = dict(
                        seed=seed,
                        steps=args.steps,
                        dimensionality=args.dimensionality,
                        conflict_intensity=conflict,
                        noise_level=noise,
                        batch_size=batch_size,
                    )
                    neat = run_synthetic(
                        config=NEATConfig(
                            learning_rate=args.learning_rate, native="never"
                        ),
                        **shared,
                    )
                    baseline = run_synthetic(
                        config=NEATConfig(
                            learning_rate=args.learning_rate,
                            nce_mode="off",
                            native="never",
                        ),
                        **shared,
                    )
                    rows.append(
                        {
                            "batch_size": batch_size,
                            "noise_level": noise,
                            "conflict_intensity": conflict,
                            "seed": seed,
                            "neat_distance": neat["final_distance_to_optimum"],
                            "baseline_distance": baseline["final_distance_to_optimum"],
                            "baseline_minus_neat": baseline["final_distance_to_optimum"]
                            - neat["final_distance_to_optimum"],
                        }
                    )
    output = Path(args.output_dir)
    write_json(
        output / "regime_map.json",
        {
            "config": vars(args),
            "rows": rows,
            "preliminary": len(args.seeds) == 1,
            "reproducibility": reproducibility_metadata(seed=args.seeds[0]),
        },
    )
    try:
        import matplotlib.pyplot as plt

        from benchmarks.style import apply_style
    except ImportError:
        return
    apply_style()
    for batch_size in args.batch_sizes:
        matrix = np.empty(
            (len(args.noise_levels), len(args.conflict_intensities)),
            dtype=np.float64,
        )
        for noise_index, noise in enumerate(args.noise_levels):
            for conflict_index, conflict in enumerate(args.conflict_intensities):
                values = [
                    row["baseline_minus_neat"]
                    for row in rows
                    if row["batch_size"] == batch_size
                    and row["noise_level"] == noise
                    and row["conflict_intensity"] == conflict
                ]
                matrix[noise_index, conflict_index] = np.mean(values)
        figure, axis = plt.subplots()
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu", origin="lower")
        axis.set(
            xlabel="Conflict intensity index",
            ylabel="Noise level index",
            title=f"Batch size {batch_size}",
        )
        figure.colorbar(image, ax=axis, label="Baseline distance - NEAT distance")
        figure.savefig(output / f"regime_map_batch{batch_size}.pdf")
        plt.close(figure)


if __name__ == "__main__":
    main()
