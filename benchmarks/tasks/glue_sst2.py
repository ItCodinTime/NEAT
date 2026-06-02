"""Reproducible Keras optimizer benchmark on GLUE SST-2."""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import keras
import numpy as np
import tensorflow as tf

from neat_optim import NEAT

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    load_dataset = None
    HF_DATASETS_IMPORT_ERROR = exc
else:
    HF_DATASETS_IMPORT_ERROR = None

try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    tfds = None
    TFDS_IMPORT_ERROR = exc
else:
    TFDS_IMPORT_ERROR = None


@dataclass(frozen=True, slots=True)
class GlueSst2BenchmarkConfig:
    """Configuration for the SST-2 benchmark."""

    seeds: tuple[int, ...] = (7, 11, 19)
    epochs: int = 3
    batch_size: int = 128
    max_tokens: int = 20000
    sequence_length: int = 64
    embedding_dim: int = 128
    train_examples_limit: int | None = None
    validation_examples_limit: int | None = None


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Serializable optimizer specification for benchmark runs."""

    label: str
    family: str
    config: dict[str, Any] = field(default_factory=dict)


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


def _require_tfds() -> None:
    if tfds is None:
        raise RuntimeError(
            "tensorflow-datasets is required for the GLUE SST-2 benchmark. "
            "Install it in the benchmark environment before running this task."
        ) from TFDS_IMPORT_ERROR


def _require_sst2_loader() -> None:
    if load_dataset is not None or tfds is not None:
        return
    raise RuntimeError(
        "GLUE SST-2 benchmark requires either `datasets` or `tensorflow-datasets` "
        "in the benchmark environment."
    ) from (HF_DATASETS_IMPORT_ERROR or TFDS_IMPORT_ERROR)


def _set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)


def _tensor_to_strings(value: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in value.reshape(-1)
        ],
        dtype=object,
    )


def _load_split(split: str, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if load_dataset is not None:
        dataset = load_dataset("glue", "sst2", split=split)
        sentences = np.asarray(dataset["sentence"], dtype=object)
        labels = np.asarray(dataset["label"], dtype=np.int32)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        return sentences, labels

    _require_tfds()
    dataset = tfds.load("glue/sst2", split=split, batch_size=-1)
    batch = tfds.as_numpy(dataset)
    sentences = _tensor_to_strings(batch["sentence"])
    labels = batch["label"].astype(np.int32)
    if limit is not None:
        sentences = sentences[:limit]
        labels = labels[:limit]
    return sentences, labels


def _load_sst2_data(config: GlueSst2BenchmarkConfig) -> dict[str, np.ndarray]:
    _require_sst2_loader()
    x_train, y_train = _load_split("train", config.train_examples_limit)
    x_val, y_val = _load_split("validation", config.validation_examples_limit)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        # GLUE test labels are not public; validation is the standard public eval split.
        "x_test": x_val,
        "y_test": y_val,
    }


def _build_model(
    config: GlueSst2BenchmarkConfig,
    train_text: np.ndarray,
) -> keras.Model:
    vectorizer = keras.layers.TextVectorization(
        max_tokens=config.max_tokens,
        output_mode="int",
        output_sequence_length=config.sequence_length,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))

    inputs = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(inputs)
    x = keras.layers.Embedding(config.max_tokens, config.embedding_dim)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_default_optimizer_specs() -> tuple[OptimizerSpec, ...]:
    """Return the baseline optimizer set plus the tuned adaptive NEAT."""
    return (
        OptimizerSpec("sgd_momentum", "sgd_momentum", {"learning_rate": 1e-2}),
        OptimizerSpec("adam", "adam", {"learning_rate": 1e-3}),
        OptimizerSpec(
            "neat_adaptive_best",
            "neat",
            {
                "learning_rate": 8e-3,
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
    config: GlueSst2BenchmarkConfig,
) -> TrialResult:
    _set_seed(seed)
    tf.keras.backend.clear_session()
    model = _build_model(config, data["x_train"])
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
        "dataset": "glue_sst2",
    }


def run_optimizer_suite(
    specs: tuple[OptimizerSpec, ...],
    config: GlueSst2BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run the SST-2 benchmark over the supplied optimizer specs."""
    config = config or GlueSst2BenchmarkConfig()
    data = _load_sst2_data(config)
    trial_results = []
    for spec in specs:
        for seed in config.seeds:
            trial_results.append(_run_trial(spec, seed, data, config))

    return {
        "task": "glue_sst2",
        "date": date.today().isoformat(),
        "environment": _benchmark_environment(),
        "config": asdict(config),
        "dataset": {
            "train_shape": [len(data["x_train"])],
            "val_shape": [len(data["x_val"])],
            "test_shape": [len(data["x_test"])],
        },
        "optimizers": [asdict(spec) for spec in specs],
        "trials": [asdict(trial) for trial in trial_results],
        "summary": _aggregate_results(trial_results),
        "notes": {
            "evaluation_split": (
                "GLUE SST-2 validation split used as public evaluation split"
            ),
        },
    }


def run_benchmark(config: GlueSst2BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run the default SST-2 benchmark suite."""
    return run_optimizer_suite(build_default_optimizer_specs(), config=config)


def main() -> None:
    print(json.dumps(run_benchmark(), indent=2))


if __name__ == "__main__":
    main()
