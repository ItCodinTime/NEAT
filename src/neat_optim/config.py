"""Configuration objects for the NEAT update rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neat_optim.exceptions import ConfigurationError

VALID_NCE_MODES = {"projection", "cosine", "off"}
VALID_UPDATE_MODES = {"momentum", "lion"}
VALID_OPPONENT_MODES = {"mean_excluding_self", "batch_mean"}
VALID_PLAYER_REDUCTIONS = {"mean", "sum"}
VALID_OPPONENT_SOURCES = {
    "momentum",
    "previous_gradient",
    "gradient_ema",
    "blended",
}


@dataclass(frozen=True, slots=True)
class NEATConfig:
    """Serializable configuration for the NEAT update rule.

    The same configuration object is used by the NumPy reference engine and the
    functional `neat_step(...)` API. The Keras optimizer mirrors these fields
    where they are user-facing.
    """

    learning_rate: float = 1e-3
    alpha: float = 0.25
    beta: float = 0.9
    nce_mode: str = "projection"
    eps: float = 1e-8
    weight_decay: float = 0.0
    decouple_weight_decay: bool = True
    nce_clip_ratio: float = 1.0
    sparsity_l1: float = 0.0
    prune_threshold: float = 0.0
    opponent_source: str = "momentum"
    opponent_ema_decay: float = 0.9
    opponent_blend: float = 0.5
    correction_warmup_steps: int = 0
    conflict_threshold: float = 0.0
    adaptive_correction: bool = False
    adaptive_correction_decay: float = 0.9
    adaptive_correction_min_scale: float = 1.0
    adaptive_correction_max_scale: float = 3.0
    adaptive_preconditioning: bool = False
    second_moment_beta: float = 0.999
    bias_correction: bool = False
    precondition_nce: bool = True
    update_mode: str = "momentum"
    adaptive_alpha: bool = False
    adaptive_alpha_min: float = 0.0
    adaptive_alpha_max: float = 1.0
    gradient_noise_decay: float = 0.95
    gradient_centralization: bool = False
    nesterov: bool = False
    lookahead_k: int = 0
    lookahead_alpha: float = 0.5
    native: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration values eagerly."""
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
        if self.sparsity_l1 < 0:
            raise ConfigurationError("sparsity_l1 must be non-negative")
        if self.prune_threshold < 0:
            raise ConfigurationError("prune_threshold must be non-negative")
        if self.opponent_source not in VALID_OPPONENT_SOURCES:
            raise ConfigurationError(
                f"opponent_source must be one of {sorted(VALID_OPPONENT_SOURCES)}"
            )
        if not 0.0 <= self.opponent_ema_decay < 1.0:
            raise ConfigurationError("opponent_ema_decay must be in [0, 1)")
        if not 0.0 <= self.opponent_blend <= 1.0:
            raise ConfigurationError("opponent_blend must be in [0, 1]")
        if self.correction_warmup_steps < 0:
            raise ConfigurationError("correction_warmup_steps must be non-negative")
        if not 0.0 <= self.conflict_threshold <= 1.0:
            raise ConfigurationError("conflict_threshold must be in [0, 1]")
        if not 0.0 <= self.adaptive_correction_decay < 1.0:
            raise ConfigurationError("adaptive_correction_decay must be in [0, 1)")
        if self.adaptive_correction_min_scale <= 0.0:
            raise ConfigurationError(
                "adaptive_correction_min_scale must be positive"
            )
        if self.adaptive_correction_max_scale < self.adaptive_correction_min_scale:
            raise ConfigurationError(
                "adaptive_correction_max_scale must be >= adaptive_correction_min_scale"
            )
        if not 0.0 <= self.second_moment_beta < 1.0:
            raise ConfigurationError("second_moment_beta must be in [0, 1)")
        if self.update_mode not in VALID_UPDATE_MODES:
            raise ConfigurationError(
                f"update_mode must be one of {sorted(VALID_UPDATE_MODES)}"
            )
        if self.adaptive_alpha_min < 0.0:
            raise ConfigurationError("adaptive_alpha_min must be non-negative")
        if self.adaptive_alpha_max < self.adaptive_alpha_min:
            raise ConfigurationError(
                "adaptive_alpha_max must be >= adaptive_alpha_min"
            )
        if not 0.0 <= self.gradient_noise_decay < 1.0:
            raise ConfigurationError("gradient_noise_decay must be in [0, 1)")
        if self.lookahead_k < 0:
            raise ConfigurationError("lookahead_k must be non-negative")
        if not 0.0 < self.lookahead_alpha <= 1.0:
            raise ConfigurationError("lookahead_alpha must be in (0, 1]")
        if self.native not in {"auto", "never", "force"}:
            raise ConfigurationError("native must be one of 'auto', 'never', 'force'")

    def as_dict(self) -> dict[str, float | bool | str]:
        """Return a plain serializable mapping of configuration fields."""
        return asdict(self)

    def as_keras_kwargs(self) -> dict[str, float | bool | str]:
        """Return keyword arguments suitable for constructing `NEAT`."""
        return self.as_dict()


@dataclass(frozen=True, slots=True)
class PlayerNEATConfig(NEATConfig):
    """Configuration for explicit per-player or per-example NEAT updates.

    Each player contributes its own gradient tensor. The optimizer constructs an
    opponent signal from the remaining players and applies a Nash-inspired
    correction before aggregating the batch update.
    """

    opponent_mode: str = "mean_excluding_self"
    player_reduction: str = "mean"
    native: str = "never"

    def __post_init__(self) -> None:
        """Validate configuration values for player-aware stepping."""
        NEATConfig.__post_init__(self)
        if self.opponent_mode not in VALID_OPPONENT_MODES:
            raise ConfigurationError(
                f"opponent_mode must be one of {sorted(VALID_OPPONENT_MODES)}"
            )
        if self.player_reduction not in VALID_PLAYER_REDUCTIONS:
            raise ConfigurationError(
                f"player_reduction must be one of {sorted(VALID_PLAYER_REDUCTIONS)}"
            )
        if self.native != "never":
            raise ConfigurationError(
                "Player-aware NEAT currently requires native='never'."
            )
