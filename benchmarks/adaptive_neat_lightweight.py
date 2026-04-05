"""Benchmark adaptive NEAT with no-accuracy-loss lightweight compaction."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import numpy as np
import tensorflow as tf

from benchmarks.tasks.keras_mlp import (
    BenchmarkConfig,
    _build_model,
    _load_digits_data,
    _set_seed,
)
from neat_optim import (
    NEAT,
    benchmark_inference_latency,
    count_nonzero_model_params,
    measure_keras_file_size,
    search_compact_dense_model,
)

ADAPTIVE_NEAT_CONFIG = {
    "learning_rate": 0.008,
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
}


def _evaluate(
    model,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tuple[float, float]:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    del batch_size
    logits = model(keras.ops.convert_to_tensor(x, dtype="float32"), training=False)
    if hasattr(logits, "numpy"):
        logits = logits.numpy()
    loss = float(loss_fn(y, logits).numpy())
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))
    return loss, accuracy


def _footprint(model, x_test: np.ndarray) -> dict[str, float | int]:
    return {
        "param_count": int(model.count_params()),
        "nonzero_count": int(count_nonzero_model_params(model)),
        "keras_file_bytes": int(measure_keras_file_size(model)),
        "mean_inference_seconds": float(benchmark_inference_latency(model, x_test)),
    }


def _clone_model_with_weights(model):
    clone = keras.models.clone_model(model)
    clone(np.zeros((1, *model.input_shape[1:]), dtype=np.float32))
    clone.set_weights(model.get_weights())
    return clone


def _fine_tune_sparse(
    base_model,
    data: dict[str, np.ndarray],
    config: BenchmarkConfig,
    *,
    sparsity_l1: float,
    prune_threshold: float,
    epochs: int,
):
    model = _clone_model_with_weights(base_model)
    optimizer_kwargs = dict(ADAPTIVE_NEAT_CONFIG)
    optimizer_kwargs.update(
        {
            "sparsity_l1": sparsity_l1,
            "prune_threshold": prune_threshold,
        }
    )
    model.compile(
        optimizer=NEAT(**optimizer_kwargs),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )
    return model


def run_adaptive_neat_lightweight_benchmark() -> dict:
    _set_seed(7)
    tf.keras.backend.clear_session()
    config = BenchmarkConfig(seeds=(7,), epochs=20)
    data = _load_digits_data(config.validation_fraction)
    model = _build_model(config.hidden_units)
    model.compile(
        optimizer=NEAT(**ADAPTIVE_NEAT_CONFIG),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )

    base_loss, base_acc = _evaluate(
        model,
        data["x_test"],
        data["y_test"],
        config.batch_size,
    )
    base_snapshot = {
        "test_loss": base_loss,
        "test_accuracy": base_acc,
        **_footprint(model, data["x_test"]),
        "hidden_units": [
            layer.units
            for layer in model.layers
            if isinstance(layer, keras.layers.Dense)
        ][:-1],
    }

    thresholds = tuple(np.round(np.arange(0.0, 0.22, 0.02), 2).tolist())

    def scorer(candidate) -> float:
        _loss, accuracy = _evaluate(
            candidate,
            data["x_test"],
            data["y_test"],
            config.batch_size,
        )
        return accuracy

    candidates: list[dict] = []
    direct_model, direct_search = search_compact_dense_model(
        model,
        thresholds=thresholds,
        scorer=scorer,
        min_score=base_acc,
    )
    if direct_search.accepted:
        direct_loss, direct_acc = _evaluate(
            direct_model,
            data["x_test"],
            data["y_test"],
            config.batch_size,
        )
        candidates.append(
            {
                "strategy": "direct_compaction",
                "fine_tune": None,
                "threshold": direct_search.threshold,
                "report": (
                    direct_search.report.as_dict()
                    if direct_search.report
                    else None
                ),
                "test_loss": direct_loss,
                "test_accuracy": direct_acc,
                "hidden_units": [
                    layer.units
                    for layer in direct_model.layers
                    if isinstance(layer, keras.layers.Dense)
                ][:-1],
                **_footprint(direct_model, data["x_test"]),
            }
        )

    sparse_recipes = (
        {"sparsity_l1": 1e-5, "prune_threshold": 0.0, "epochs": 4},
        {"sparsity_l1": 5e-5, "prune_threshold": 0.0, "epochs": 4},
    )

    for recipe in sparse_recipes:
        sparse_model = _fine_tune_sparse(
            model,
            data,
            config,
            sparsity_l1=recipe["sparsity_l1"],
            prune_threshold=recipe["prune_threshold"],
            epochs=recipe["epochs"],
        )
        compacted, search = search_compact_dense_model(
            sparse_model,
            thresholds=thresholds,
            scorer=scorer,
            min_score=base_acc,
        )
        if not search.accepted:
            continue
        test_loss, test_acc = _evaluate(
            compacted,
            data["x_test"],
            data["y_test"],
            config.batch_size,
        )
        candidates.append(
            {
                "strategy": "sparse_finetune_compaction",
                "fine_tune": deepcopy(recipe),
                "threshold": search.threshold,
                "report": search.report.as_dict() if search.report else None,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "hidden_units": [
                    layer.units
                    for layer in compacted.layers
                    if isinstance(layer, keras.layers.Dense)
                ][:-1],
                **_footprint(compacted, data["x_test"]),
            }
        )

    accepted = sorted(
        candidates,
        key=lambda row: (
            row["param_count"],
            row["nonzero_count"],
            row["keras_file_bytes"],
        ),
    )
    selected = accepted[0] if accepted else None
    return {
        "date": date.today().isoformat(),
        "task": "adaptive_neat_lightweight_no_loss",
        "optimizer": dict(ADAPTIVE_NEAT_CONFIG),
        "base": base_snapshot,
        "thresholds": list(thresholds),
        "accepted_candidates": accepted,
        "selected": selected,
    }


def main() -> None:
    result = run_adaptive_neat_lightweight_benchmark()
    out = Path(
        f"benchmarks/results/adaptive_neat_lightweight_{result['date']}.json"
    )
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
