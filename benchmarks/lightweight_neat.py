from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
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
    compact_dense_model,
    count_nonzero_model_params,
    measure_keras_file_size,
)


def run_lightweight_neat_benchmark() -> dict:
    _set_seed(7)
    tf.keras.backend.clear_session()
    config = BenchmarkConfig(seeds=(7,), epochs=20)
    data = _load_digits_data(config.validation_fraction)
    model = _build_model(config.hidden_units)
    optimizer = NEAT(
        learning_rate=0.05,
        alpha=0.25,
        beta=0.9,
        nce_mode="projection",
        sparsity_l1=1e-3,
        prune_threshold=5e-3,
    )
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])
    model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=True,
        verbose=0,
    )

    base_loss, base_acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    compacted, report = compact_dense_model(model, unit_threshold=0.16)
    compacted.compile(
        loss=loss_fn,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    compacted_loss, compacted_acc = compacted.evaluate(
        data["x_test"],
        data["y_test"],
        verbose=0,
    )

    return {
        "date": date.today().isoformat(),
        "task": "digits_lightweight_neat",
        "optimizer": {
            "learning_rate": 0.05,
            "alpha": 0.25,
            "beta": 0.9,
            "nce_mode": "projection",
            "sparsity_l1": 1e-3,
            "prune_threshold": 5e-3,
        },
        "compaction_threshold": 0.16,
        "base": {
            "test_loss": float(base_loss),
            "test_accuracy": float(base_acc),
            "param_count": int(model.count_params()),
            "nonzero_count": int(count_nonzero_model_params(model)),
            "keras_file_bytes": int(measure_keras_file_size(model)),
            "mean_inference_seconds": float(
                benchmark_inference_latency(model, data["x_test"])
            ),
            "hidden_units": [
                layer.units
                for layer in model.layers
                if isinstance(layer, keras.layers.Dense)
            ][:-1],
        },
        "compacted": {
            "test_loss": float(compacted_loss),
            "test_accuracy": float(compacted_acc),
            "param_count": int(compacted.count_params()),
            "nonzero_count": int(count_nonzero_model_params(compacted)),
            "keras_file_bytes": int(measure_keras_file_size(compacted)),
            "mean_inference_seconds": float(
                benchmark_inference_latency(compacted, data["x_test"])
            ),
            "hidden_units": [
                layer.units
                for layer in compacted.layers
                if isinstance(layer, keras.layers.Dense)
            ][:-1],
        },
        "report": report.as_dict(),
    }


def main() -> None:
    result = run_lightweight_neat_benchmark()
    out = Path(
        f"benchmarks/results/neat_lightweight_digits_{result['date']}.json"
    )
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
