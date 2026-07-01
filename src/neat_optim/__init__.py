"""Public package entrypoints for NEAT."""

from neat_optim._version import __version__
from neat_optim.config import NEATConfig, PlayerNEATConfig
from neat_optim.state import (
    ArrayState,
    PlayerStepMetrics,
    PlayerStepResult,
    StepMetrics,
    StepResult,
)

__all__ = [
    "ArrayState",
    "DenseCompactionReport",
    "DenseCompactionSearchResult",
    "NEAT",
    "NEATDiagnosticsCallback",
    "NEATConfig",
    "TorchNEAT",
    "PlayerNEATConfig",
    "PlayerStepMetrics",
    "PlayerStepResult",
    "StepMetrics",
    "StepResult",
    "__version__",
    "benchmark_inference_latency",
    "compact_dense_model",
    "count_nonzero_model_params",
    "measure_keras_file_size",
    "search_compact_dense_model",
]


def __getattr__(name: str):
    if name == "TorchNEAT":
        try:
            from neat_optim.torch_optimizer import TorchNEAT
        except Exception as exc:  # pragma: no cover - import guard behavior
            raise ImportError(
                "TorchNEAT requires PyTorch. Install `neat-optim[torch]`."
            ) from exc
        return TorchNEAT
    if name == "NEAT":
        try:
            from neat_optim.keras_optimizer import NEAT
        except Exception as exc:  # pragma: no cover - import guard behavior
            raise ImportError(
                "The Keras optimizer requires the optional Keras runtime. "
                "Install `neat-optim[keras]` and a supported backend such as "
                "TensorFlow before importing `NEAT`."
            ) from exc
        return NEAT
    if name == "NEATDiagnosticsCallback":
        try:
            from neat_optim.callbacks import NEATDiagnosticsCallback
        except Exception as exc:  # pragma: no cover - import guard behavior
            raise ImportError(
                "NEAT diagnostics callbacks require the optional Keras runtime. "
                "Install `neat-optim[keras]` and a supported backend such as "
                "TensorFlow before importing `NEATDiagnosticsCallback`."
            ) from exc
        return NEATDiagnosticsCallback
    if name in {
        "DenseCompactionReport",
        "DenseCompactionSearchResult",
        "benchmark_inference_latency",
        "compact_dense_model",
        "count_nonzero_model_params",
        "measure_keras_file_size",
        "search_compact_dense_model",
    }:
        try:
            from neat_optim.compaction import (
                DenseCompactionReport,
                DenseCompactionSearchResult,
                benchmark_inference_latency,
                compact_dense_model,
                count_nonzero_model_params,
                measure_keras_file_size,
                search_compact_dense_model,
            )
        except Exception as exc:  # pragma: no cover - import guard behavior
            raise ImportError(
                "Model compaction utilities require the optional Keras runtime. "
                "Install `neat-optim[keras]` and a supported backend such as "
                "TensorFlow before importing compaction helpers."
            ) from exc
        exports = {
            "DenseCompactionReport": DenseCompactionReport,
            "DenseCompactionSearchResult": DenseCompactionSearchResult,
            "benchmark_inference_latency": benchmark_inference_latency,
            "compact_dense_model": compact_dense_model,
            "count_nonzero_model_params": count_nonzero_model_params,
            "measure_keras_file_size": measure_keras_file_size,
            "search_compact_dense_model": search_compact_dense_model,
        }
        return exports[name]
    raise AttributeError(f"module 'neat_optim' has no attribute {name!r}")
