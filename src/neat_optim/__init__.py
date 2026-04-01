"""Public package entrypoints for NEAT."""

from neat_optim._version import __version__
from neat_optim.config import NEATConfig
from neat_optim.state import ArrayState, StepMetrics, StepResult

__all__ = [
    "ArrayState",
    "NEAT",
    "NEATConfig",
    "StepMetrics",
    "StepResult",
    "__version__",
]


def __getattr__(name: str):
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
    raise AttributeError(f"module 'neat_optim' has no attribute {name!r}")
