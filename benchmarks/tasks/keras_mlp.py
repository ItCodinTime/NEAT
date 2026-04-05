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

from neat_optim import NEAT, PlayerNEATConfig
from neat_optim.training import create_player_states, player_train_step


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration for the digits MLP benchmark."""

    seeds: tuple[int, ...] = (7, 11, 19)
    epochs: int = 20
    batch_size: int = 64
    hidden_units: tuple[int, int] = (128, 64)
    validation_fraction: float = 0.2
    player_batch_size: int = 32


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
        "blended",
    )
    opponent_blends: tuple[float, ...] = (0.25, 0.5)
    correction_warmup_steps: tuple[int, ...] = (0,)
    conflict_thresholds: tuple[float, ...] = (0.0,)
    adaptive_corrections: tuple[bool, ...] = (False, True)
    adaptive_correction_decays: tuple[float, ...] = (0.9,)
    adaptive_correction_max_scales: tuple[float, ...] = (2.0, 3.0)
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
        OptimizerSpec(
            "player_neat",
            "player_neat",
            {
                "learning_rate": 6e-2,
                "alpha": 0.25,
                "beta": 0.9,
                "nce_mode": "projection",
                "nce_clip_ratio": 1.0,
                "adaptive_correction": True,
                "adaptive_correction_decay": 0.9,
                "adaptive_correction_max_scale": 2.5,
                "opponent_mode": "mean_excluding_self",
                "player_reduction": "mean",
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


def _evaluate_model(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    logits = model.predict(x, batch_size=batch_size, verbose=0)
    loss = float(loss_fn(y, logits).numpy())
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))
    return loss, accuracy


def _mean_player_metrics(
    metrics_rows: list[dict[str, float]],
) -> dict[str, float | None]:
    if not metrics_rows:
        return {
            "mean_conflict_ratio": None,
            "mean_correction_ratio": None,
            "mean_update_alignment": None,
            "mean_opponent_norm": None,
            "correction_active_fraction": None,
        }
    return {
        "mean_conflict_ratio": float(
            np.mean([row["mean_player_conflict"] for row in metrics_rows])
        ),
        "mean_correction_ratio": float(
            np.mean([row["mean_correction_ratio"] for row in metrics_rows])
        ),
        "mean_update_alignment": None,
        "mean_opponent_norm": None,
        "correction_active_fraction": None,
    }


def _run_player_trial(
    spec: OptimizerSpec,
    seed: int,
    data: dict[str, np.ndarray],
    config: BenchmarkConfig,
) -> TrialResult:
    _set_seed(seed)
    tf.keras.backend.clear_session()
    model = _build_model(config.hidden_units)
    _ = model(data["x_train"][:1], training=False)
    states = create_player_states(model)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )
    player_config = PlayerNEATConfig(native="never", **spec.config)

    rng = np.random.default_rng(seed)
    batch_size = config.player_batch_size
    metrics_rows: list[dict[str, float]] = []
    start = time.perf_counter()
    for _epoch in range(config.epochs):
        indices = rng.permutation(len(data["x_train"]))
        for start_index in range(0, len(indices), batch_size):
            batch_indices = indices[start_index : start_index + batch_size]
            batch_x = tf.convert_to_tensor(data["x_train"][batch_indices])
            batch_y = tf.convert_to_tensor(data["y_train"][batch_indices])
            result = player_train_step(
                model,
                batch_x,
                batch_y,
                loss_fn,
                states,
                player_config,
            )
            states = result.states
            if result.metrics:
                metrics_rows.append(
                    {
                        "mean_player_conflict": float(
                            np.mean(
                                [
                                    metric.mean_player_conflict
                                    for metric in result.metrics
                                ]
                            )
                        ),
                        "mean_correction_ratio": float(
                            np.mean(
                                [
                                    metric.mean_correction_ratio
                                    for metric in result.metrics
                                ]
                            )
                        ),
                    }
                )
    elapsed = time.perf_counter() - start

    train_loss, train_accuracy = _evaluate_model(
        model,
        data["x_train"],
        data["y_train"],
        batch_size=config.batch_size,
    )
    val_loss, val_accuracy = _evaluate_model(
        model,
        data["x_val"],
        data["y_val"],
        batch_size=config.batch_size,
    )
    test_loss, test_accuracy = _evaluate_model(
        model,
        data["x_test"],
        data["y_test"],
        batch_size=config.batch_size,
    )
    diagnostics = _mean_player_metrics(metrics_rows)
    return TrialResult(
        optimizer=spec.label,
        optimizer_family=spec.family,
        optimizer_config=dict(spec.config),
        seed=seed,
        epochs=config.epochs,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        seconds=float(elapsed),
        mean_conflict_ratio=diagnostics["mean_conflict_ratio"],
        mean_correction_ratio=diagnostics["mean_correction_ratio"],
        mean_update_alignment=diagnostics["mean_update_alignment"],
        mean_opponent_norm=diagnostics["mean_opponent_norm"],
        correction_active_fraction=diagnostics["correction_active_fraction"],
    )


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
    if spec.family == "player_neat":
        return _run_player_trial(spec, seed, data, config)

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
        f"_blend={config['opponent_blend']}"
        f"_warmup={config['correction_warmup_steps']}"
        f"_threshold={config['conflict_threshold']}"
        f"_adaptive={int(config['adaptive_correction'])}"
        f"_amax={config['adaptive_correction_max_scale']}"
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
        opponent_blend,
        correction_warmup_steps,
        conflict_threshold,
        adaptive_correction,
        adaptive_correction_decay,
        adaptive_correction_max_scale,
    ) in product(
        search.learning_rates,
        search.alphas,
        search.betas,
        search.nce_modes,
        search.nce_clip_ratios,
        search.opponent_sources,
        search.opponent_blends,
        search.correction_warmup_steps,
        search.conflict_thresholds,
        search.adaptive_corrections,
        search.adaptive_correction_decays,
        search.adaptive_correction_max_scales,
    ):
        config = {
            "learning_rate": learning_rate,
            "alpha": alpha,
            "beta": beta,
            "nce_mode": nce_mode,
            "nce_clip_ratio": nce_clip_ratio,
            "opponent_source": opponent_source,
            "opponent_blend": opponent_blend,
            "correction_warmup_steps": correction_warmup_steps,
            "conflict_threshold": conflict_threshold,
            "adaptive_correction": adaptive_correction,
            "adaptive_correction_decay": adaptive_correction_decay,
            "adaptive_correction_min_scale": 1.0,
            "adaptive_correction_max_scale": adaptive_correction_max_scale,
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
