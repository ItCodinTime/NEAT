"""Utilities for compacting sparse Keras Dense models."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class DenseCompactionReport:
    """Summary of a Dense-model compaction pass."""

    original_hidden_units: tuple[int, ...]
    compacted_hidden_units: tuple[int, ...]
    original_param_count: int
    compacted_param_count: int
    original_nonzero_count: int
    compacted_nonzero_count: int
    unit_threshold: float

    def as_dict(self) -> dict[str, Any]:
        """Return a plain serializable mapping."""
        return asdict(self)


def _keras():
    import keras

    return keras


def _dense_layers(model) -> list[Any]:
    keras = _keras()
    if not isinstance(model, keras.Sequential):
        raise TypeError("compact_dense_model currently supports keras.Sequential only")
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
    ]
    if not dense_layers:
        raise ValueError("compact_dense_model requires at least one Dense layer")
    unsupported = [
        layer
        for layer in model.layers
        if not isinstance(layer, (keras.layers.InputLayer, keras.layers.Dense))
    ]
    if unsupported:
        names = ", ".join(type(layer).__name__ for layer in unsupported)
        raise TypeError(
            "compact_dense_model only supports InputLayer + Dense stacks; "
            f"got unsupported layers: {names}"
        )
    return dense_layers


def _activation_name(layer: Any) -> str:
    activation = getattr(layer, "activation", None)
    if activation is None:
        return "linear"
    return getattr(activation, "__name__", "unknown")


def _zero_preserving_activation(layer: Any) -> bool:
    return _activation_name(layer) in {"linear", "relu"}


def _keep_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    keep = scores > threshold
    if np.any(keep):
        return keep
    keep[np.argmax(scores)] = True
    return keep


def _weight_array(layer: Any) -> tuple[np.ndarray, np.ndarray]:
    weights = layer.get_weights()
    kernel = np.asarray(weights[0], dtype=np.float32)
    if layer.use_bias:
        bias = np.asarray(weights[1], dtype=np.float32)
    else:
        bias = np.zeros(kernel.shape[1], dtype=np.float32)
    return kernel, bias


def count_nonzero_model_params(model, threshold: float = 0.0) -> int:
    """Return the number of parameters above `threshold` in magnitude."""
    dense_layers = _dense_layers(model)
    total = 0
    for layer in dense_layers:
        for weight in layer.get_weights():
            total += int(np.count_nonzero(np.abs(weight) > threshold))
    return total


def compact_dense_model(
    model,
    *,
    unit_threshold: float = 0.0,
) -> tuple[Any, DenseCompactionReport]:
    """Return a smaller Sequential Dense model by removing weak hidden units.

    Hidden units are retained when they still have outgoing signal above
    `unit_threshold`. For zero-preserving activations such as ReLU, units that
    are already dead from zeroed incoming weights and bias are also removed.
    """

    keras = _keras()
    dense_layers = _dense_layers(model)
    kernels: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for layer in dense_layers:
        kernel, bias = _weight_array(layer)
        kernels.append(kernel)
        biases.append(bias)

    keep_masks: list[np.ndarray] = [
        np.ones(layer.units, dtype=bool) for layer in dense_layers
    ]
    for index in range(len(dense_layers) - 2, -1, -1):
        next_kernel = kernels[index + 1]
        next_keep = keep_masks[index + 1]
        outgoing_strength = np.max(np.abs(next_kernel[:, next_keep]), axis=1)
        keep = _keep_mask(outgoing_strength, unit_threshold)
        if _zero_preserving_activation(dense_layers[index]):
            incoming_strength = np.max(np.abs(kernels[index]), axis=0)
            bias_strength = np.abs(biases[index])
            alive = (incoming_strength > unit_threshold) | (
                bias_strength > unit_threshold
            )
            keep = keep & alive
            if not np.any(keep):
                importance = np.maximum(
                    outgoing_strength,
                    np.maximum(incoming_strength, bias_strength),
                )
                keep[np.argmax(importance)] = True
        keep_masks[index] = keep

    compacted = keras.Sequential(name=f"{model.name}_compacted")
    input_shape = tuple(int(dim) for dim in model.input_shape[1:])
    compacted.add(keras.layers.Input(shape=input_shape))

    compacted_hidden_units: list[int] = []
    for index, layer in enumerate(dense_layers):
        column_keep = keep_masks[index]
        units = int(np.count_nonzero(column_keep))
        if index < len(dense_layers) - 1:
            compacted_hidden_units.append(units)
        compacted.add(
            keras.layers.Dense(
                units,
                activation=layer.activation,
                use_bias=layer.use_bias,
                name=f"{layer.name}_compacted",
                dtype=layer.dtype_policy.name,
            )
        )
    compacted(np.zeros((1, *input_shape), dtype=np.float32))

    previous_keep = np.ones(kernels[0].shape[0], dtype=bool)
    compacted_layers = [
        layer for layer in compacted.layers if isinstance(layer, keras.layers.Dense)
    ]
    for index, layer in enumerate(dense_layers):
        column_keep = keep_masks[index]
        kernel = kernels[index][np.ix_(previous_keep, column_keep)]
        if layer.use_bias:
            compacted_layers[index].set_weights([kernel, biases[index][column_keep]])
        else:
            compacted_layers[index].set_weights([kernel])
        previous_keep = column_keep

    report = DenseCompactionReport(
        original_hidden_units=tuple(layer.units for layer in dense_layers[:-1]),
        compacted_hidden_units=tuple(compacted_hidden_units),
        original_param_count=int(model.count_params()),
        compacted_param_count=int(compacted.count_params()),
        original_nonzero_count=count_nonzero_model_params(model),
        compacted_nonzero_count=count_nonzero_model_params(compacted),
        unit_threshold=float(unit_threshold),
    )
    return compacted, report


def measure_keras_file_size(model) -> int:
    """Return the on-disk size of an uncompiled `.keras` export in bytes."""
    keras = _keras()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.keras"
        clone = keras.models.clone_model(model)
        clone(np.zeros((1, *model.input_shape[1:]), dtype=np.float32))
        clone.set_weights(model.get_weights())
        clone.save(path)
        return int(os.stat(path).st_size)


def benchmark_inference_latency(
    model,
    inputs: np.ndarray,
    *,
    warmup_runs: int = 5,
    timed_runs: int = 25,
) -> float:
    """Return average inference latency in seconds for full-batch forward passes."""
    batch = _keras().ops.convert_to_tensor(inputs, dtype="float32")
    for _ in range(warmup_runs):
        output = model(batch, training=False)
        if hasattr(output, "numpy"):
            output.numpy()
    start = time.perf_counter()
    for _ in range(timed_runs):
        output = model(batch, training=False)
        if hasattr(output, "numpy"):
            output.numpy()
    return float((time.perf_counter() - start) / timed_runs)
