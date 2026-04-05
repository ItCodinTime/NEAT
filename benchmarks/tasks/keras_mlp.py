"""Reproducible Keras optimizer benchmark on the sklearn digits dataset."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import product
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import keras
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neat_optim import NEAT


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for the digits MLP benchmark."""

    seeds: tuple[int, ...] = (7, 11, 19)
    epochs: int = 20
    batch_size: int = 64
    hidden_units: tuple[int, int] = (128, 64)
    validation_fraction: float = 0.2


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Serializable optimizer specification for benchmark runs."""

    label: str
    family: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NEATSweepConfig:
    """Search space for NEAT hyperparameter sweeps."""

    learning_rates: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)
    alphas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    betas: tuple[float, ...] = (0.9, 0.95)
    nce_modes: tuple[str, ...] = ("projection", "off")
    nce_clip_ratios: tuple[float, ...] = (1.0,)
    opponent_sources: tuple[str, ...] = (
        "momentum",
        "previous_gradient",
        "gradient_ema",
    )
    correction_warmup_steps: tuple[int, ...] = (0,)
    conflict_thresholds: tuple[float, ...] = (0.0,)
    seeds: tuple[int, ...] = (7,)
    top_k: int = 10


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Single-seed benchmark result."""

    optimizer: str
    optimizer_family: str
    optimizer_config: dict[str, Any]
    seed: int
    epochs: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float
    seconds: float
    mean_conflict_ratio: float | None = None
    mean_correction_ratio: float | None = None
    mean_update_alignment: float | None = None
    mean_opponent_norm: float | None = None
    correction_active_fraction: float | None = None


def _set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)


def _load_digits_data(validation_fraction: float) -> dict[str, np.ndarray]:
    x, y = load_digits(return_X_y=True)
    x = x.astype(np.float32)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    val_size = validation_fraction / 0.8
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=42,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return {
        "x_train": x_train,
        "y_train": y_train.astype(np.int32),
        "x_val": x_val,
        "y_val": y_val.astype(np.int32),
        "x_test": x_test,
        "y_test": y_test.astype(np.int32),
    }


def _build_model(hidden_units: tuple[int, int]) -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.Input((64,)),
            keras.layers.Dense(hidden_units[0], activation="relu"),
            keras.layers.Dense(hidden_units[1], activation="relu"),
            keras.layers.Dense(10),
        ]
    )


def build_default_optimizer_specs() -> tuple[OptimizerSpec, ...]:
    """Return the baseline optimizer set plus the tuned NEAT default."""
    return (
        OptimizerSpec("sgd_momentum", "sgd_momentum", {"learning_rate": 1e-2}),
        OptimizerSpec("adam", "adam", {"learning_rate": 1e-3}),
        OptimizerSpec("adamw", "adamw", {"learning_rate": 1e-3, "weight_decay": 1e-4}),
        OptimizerSpec(
            "neat",
            "neat",
            {
                "learning_rate": 3e-2,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "opponent_source": "momentum",
                "correction_warmup_steps": 0,
                "conflict_threshold": 0.0,
            },
        ),
    )


def _build_optimizer(spec: OptimizerSpec) -> Any:
    if spec.family == "sgd_momentum":
        return keras.optimizers.SGD(
            learning_rate=spec.config.get("learning_rate", 1e-2),
            momentum=0.9,
        )
    if spec.family == "adam":
        return keras.optimizers.Adam(
            learning_rate=spec.config.get("learning_rate", 1e-3)
        )
    if spec.family == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=spec.config.get("learning_rate", 1e-3),
            weight_decay=spec.config.get("weight_decay", 1e-4),
        )
    if spec.family == "neat":
        return NEAT(**spec.config)
    raise ValueError(f"unknown optimizer family: {spec.family}")


def _trial_diagnostics(optimizer: Any) -> dict[str, float | None]:
    if not hasattr(optimizer, "diagnostic_snapshot"):
        return {
            "mean_conflict_ratio": None,
            "mean_correction_ratio": None,
            "mean_update_alignment": None,
            "mean_opponent_norm": None,
            "correction_active_fraction": None,
        }
    snapshot = optimizer.diagnostic_snapshot()
    if not snapshot:
        return {
            "mean_conflict_ratio": None,
            "mean_correction_ratio": None,
            "mean_update_alignment": None,
            "mean_opponent_norm": None,
            "correction_active_fraction": None,
        }
    return snapshot


def _run_trial(
    spec: OptimizerSpec,
    seed: int,
    data: dict[str, np.ndarray],
    config: BenchmarkConfig,
) -> TrialResult:
    _set_seed(seed)
    tf.keras.backend.clear_session()
    model = _build_model(config.hidden_units)
    optimizer = _build_optimizer(spec)
    if hasattr(optimizer, "reset_diagnostics"):
        optimizer.reset_diagnostics()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    start = time.perf_counter()
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )
    elapsed = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(
        data["x_test"],
        data["y_test"],
        batch_size=config.batch_size,
        verbose=0,
    )
    diagnostics = _trial_diagnostics(optimizer)

    return TrialResult(
        optimizer=spec.label,
        optimizer_family=spec.family,
        optimizer_config=dict(spec.config),
        seed=seed,
        epochs=config.epochs,
        train_loss=float(history.history["loss"][-1]),
        train_accuracy=float(history.history["accuracy"][-1]),
        val_loss=float(history.history["val_loss"][-1]),
        val_accuracy=float(history.history["val_accuracy"][-1]),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        seconds=float(elapsed),
        mean_conflict_ratio=diagnostics["mean_conflict_ratio"],
        mean_correction_ratio=diagnostics["mean_correction_ratio"],
        mean_update_alignment=diagnostics["mean_update_alignment"],
        mean_opponent_norm=diagnostics["mean_opponent_norm"],
        correction_active_fraction=diagnostics["correction_active_fraction"],
    )


def _mean_optional(values: list[float | None]) -> float | None:
    selected = [value for value in values if value is not None]
    if not selected:
        return None
    return float(np.mean(selected))


def _aggregate_results(trials: list[TrialResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = sorted({trial.optimizer for trial in trials})
    for label in labels:
        selected = [trial for trial in trials if trial.optimizer == label]
        rows.append(
            {
                "optimizer": label,
                "optimizer_family": selected[0].optimizer_family,
                "optimizer_config": dict(selected[0].optimizer_config),
                "num_trials": len(selected),
                "mean_test_accuracy": float(
                    np.mean([trial.test_accuracy for trial in selected])
                ),
                "std_test_accuracy": float(
                    np.std([trial.test_accuracy for trial in selected])
                ),
                "mean_test_loss": float(
                    np.mean([trial.test_loss for trial in selected])
                ),
                "mean_val_accuracy": float(
                    np.mean([trial.val_accuracy for trial in selected])
                ),
                "mean_seconds": float(np.mean([trial.seconds for trial in selected])),
                "mean_conflict_ratio": _mean_optional(
                    [trial.mean_conflict_ratio for trial in selected]
                ),
                "mean_correction_ratio": _mean_optional(
                    [trial.mean_correction_ratio for trial in selected]
                ),
                "mean_update_alignment": _mean_optional(
                    [trial.mean_update_alignment for trial in selected]
                ),
                "mean_opponent_norm": _mean_optional(
                    [trial.mean_opponent_norm for trial in selected]
                ),
                "correction_active_fraction": _mean_optional(
                    [trial.correction_active_fraction for trial in selected]
                ),
            }
        )
    rows.sort(key=lambda row: float(row["mean_test_accuracy"]), reverse=True)
    return rows


def _benchmark_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "dataset": "sklearn_digits",
    }


def run_optimizer_suite(
    specs: tuple[OptimizerSpec, ...],
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a benchmark suite over the supplied optimizer specs."""
    config = config or BenchmarkConfig()
    data = _load_digits_data(config.validation_fraction)
    trial_results = []
    for spec in specs:
        for seed in config.seeds:
            trial_results.append(_run_trial(spec, seed, data, config))

    return {
        "task": "keras_mlp_digits",
        "date": date.today().isoformat(),
        "environment": _benchmark_environment(),
        "config": asdict(config),
        "dataset": {
            "train_shape": list(data["x_train"].shape),
            "val_shape": list(data["x_val"].shape),
            "test_shape": list(data["x_test"].shape),
        },
        "optimizers": [asdict(spec) for spec in specs],
        "trials": [asdict(trial) for trial in trial_results],
        "summary": _aggregate_results(trial_results),
    }


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run the default benchmark suite and return machine-readable results."""
    return run_optimizer_suite(build_default_optimizer_specs(), config=config)


def _sweep_label(config: dict[str, Any]) -> str:
    return (
        "neat"
        f"_lr={config['learning_rate']}"
        f"_a={config['alpha']}"
        f"_b={config['beta']}"
        f"_mode={config['nce_mode']}"
        f"_clip={config['nce_clip_ratio']}"
        f"_opp={config['opponent_source']}"
        f"_warmup={config['correction_warmup_steps']}"
        f"_threshold={config['conflict_threshold']}"
    )


def build_neat_sweep_specs(search: NEATSweepConfig) -> tuple[OptimizerSpec, ...]:
    """Expand a NEAT search space into optimizer specs."""
    specs = []
    for (
        learning_rate,
        alpha,
        beta,
        nce_mode,
        nce_clip_ratio,
        opponent_source,
        correction_warmup_steps,
        conflict_threshold,
    ) in product(
        search.learning_rates,
        search.alphas,
        search.betas,
        search.nce_modes,
        search.nce_clip_ratios,
        search.opponent_sources,
        search.correction_warmup_steps,
        search.conflict_thresholds,
    ):
        config = {
            "learning_rate": learning_rate,
            "alpha": alpha,
            "beta": beta,
            "nce_mode": nce_mode,
            "nce_clip_ratio": nce_clip_ratio,
            "opponent_source": opponent_source,
            "correction_warmup_steps": correction_warmup_steps,
            "conflict_threshold": conflict_threshold,
        }
        specs.append(OptimizerSpec(_sweep_label(config), "neat", config))
    return tuple(specs)


def run_neat_sweep(
    search: NEATSweepConfig | None = None,
    benchmark: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a NEAT-only sweep and rank configurations by validation accuracy."""
    search = search or NEATSweepConfig()
    benchmark = benchmark or BenchmarkConfig(seeds=search.seeds)
    suite = run_optimizer_suite(build_neat_sweep_specs(search), config=benchmark)
    ranked = sorted(
        suite["summary"],
        key=lambda row: (
            float(row["mean_val_accuracy"]),
            float(row["mean_test_accuracy"]),
        ),
        reverse=True,
    )
    suite["search_space"] = asdict(search)
    suite["top_configs"] = ranked[: search.top_k]
    return suite


def main() -> None:
    print(json.dumps(run_benchmark(), indent=2))


if __name__ == "__main__":
    main()
