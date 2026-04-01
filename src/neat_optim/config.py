from __future__ import annotations

from dataclasses import asdict, dataclass

from neat_optim.exceptions import ConfigurationError

VALID_NCE_MODES = {"projection", "cosine", "off"}


@dataclass(frozen=True, slots=True)
class NEATConfig:
    """Serializable configuration for the NEAT update rule."""

    learning_rate: float = 1e-3
    alpha: float = 0.25
    beta: float = 0.9
    nce_mode: str = "projection"
    eps: float = 1e-8
    weight_decay: float = 0.0
    decouple_weight_decay: bool = True
    nce_clip_ratio: float = 1.0
    native: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")
        if self.alpha < 0.0:
            raise ConfigurationError("alpha must be non-negative")
        if not 0.0 <= self.beta < 1.0:
            raise ConfigurationError("beta must be in [0, 1)")
        if self.nce_mode not in VALID_NCE_MODES:
            raise ConfigurationError(
                f"nce_mode must be one of {sorted(VALID_NCE_MODES)}"
            )
        if self.eps <= 0:
            raise ConfigurationError("eps must be positive")
        if self.weight_decay < 0:
            raise ConfigurationError("weight_decay must be non-negative")
        if self.nce_clip_ratio <= 0:
            raise ConfigurationError("nce_clip_ratio must be positive")
        if self.native not in {"auto", "never", "force"}:
            raise ConfigurationError("native must be one of 'auto', 'never', 'force'")

    def as_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)

    def as_keras_kwargs(self) -> dict[str, float | bool | str]:
        return self.as_dict()
