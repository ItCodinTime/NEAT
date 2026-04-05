from __future__ import annotations

from typing import Any

import keras
from keras import ops

from neat_optim.config import NEATConfig


@keras.saving.register_keras_serializable(package="neat_optim")
class NEAT(keras.optimizers.Optimizer):
    """Keras optimizer implementation of Nash-Equilibrium Adaptive Training."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        alpha: float = 0.25,
        beta: float = 0.9,
        nce_mode: str = "projection",
        eps: float = 1e-8,
        weight_decay: float | None = 0.0,
        decouple_weight_decay: bool = True,
        nce_clip_ratio: float = 1.0,
        sparsity_l1: float = 0.0,
        prune_threshold: float = 0.0,
        opponent_source: str = "momentum",
        opponent_ema_decay: float = 0.9,
        opponent_blend: float = 0.5,
        correction_warmup_steps: int = 0,
        conflict_threshold: float = 0.0,
        adaptive_correction: bool = False,
        adaptive_correction_decay: float = 0.9,
        adaptive_correction_min_scale: float = 1.0,
        adaptive_correction_max_scale: float = 3.0,
        name: str = "NEAT",
        **kwargs: Any,
    ) -> None:
        config = NEATConfig(
            learning_rate=learning_rate,
            alpha=alpha,
            beta=beta,
            nce_mode=nce_mode,
            eps=eps,
            weight_decay=0.0 if weight_decay is None else weight_decay,
            decouple_weight_decay=decouple_weight_decay,
            nce_clip_ratio=nce_clip_ratio,
            sparsity_l1=sparsity_l1,
            prune_threshold=prune_threshold,
            opponent_source=opponent_source,
            opponent_ema_decay=opponent_ema_decay,
            opponent_blend=opponent_blend,
            correction_warmup_steps=correction_warmup_steps,
            conflict_threshold=conflict_threshold,
            adaptive_correction=adaptive_correction,
            adaptive_correction_decay=adaptive_correction_decay,
            adaptive_correction_min_scale=adaptive_correction_min_scale,
            adaptive_correction_max_scale=adaptive_correction_max_scale,
            native="never",
        )
        super().__init__(
            learning_rate=learning_rate,
            # NEAT applies weight decay explicitly inside update_step so the
            # coupled and decoupled paths stay aligned with the reference
            # engine instead of also triggering Keras base-class decay.
            weight_decay=None,
            name=name,
            **kwargs,
        )
        self.alpha = config.alpha
        self.beta = config.beta
        self.nce_mode = config.nce_mode
        self.eps = config.eps
        self.decouple_weight_decay = config.decouple_weight_decay
        self.neat_weight_decay = config.weight_decay
        self.nce_clip_ratio = config.nce_clip_ratio
        self.sparsity_l1 = config.sparsity_l1
        self.prune_threshold = config.prune_threshold
        self.opponent_source = config.opponent_source
        self.opponent_ema_decay = config.opponent_ema_decay
        self.opponent_blend = config.opponent_blend
        self.correction_warmup_steps = config.correction_warmup_steps
        self.conflict_threshold = config.conflict_threshold
        self.adaptive_correction = config.adaptive_correction
        self.adaptive_correction_decay = config.adaptive_correction_decay
        self.adaptive_correction_min_scale = config.adaptive_correction_min_scale
        self.adaptive_correction_max_scale = config.adaptive_correction_max_scale
        self.spec_version = "nce_spec_v1"
        self.momentums = []
        self.nces = []
        self.previous_gradients = []
        self.gradient_emas = []
        self.conflict_emas = []
        self.diagnostic_conflict_sum = None
        self.diagnostic_correction_ratio_sum = None
        self.diagnostic_update_alignment_sum = None
        self.diagnostic_opponent_norm_sum = None
        self.diagnostic_correction_active_sum = None
        self.diagnostic_count = None

    def build(self, variables) -> None:
        if self.built:
            return
        super().build(variables)
        self.momentums = self.add_optimizer_variables(variables, "momentum")
        self.nces = self.add_optimizer_variables(variables, "nce")
        self.previous_gradients = self.add_optimizer_variables(variables, "prev_grad")
        self.gradient_emas = self.add_optimizer_variables(variables, "grad_ema")
        self.conflict_emas = self.add_optimizer_variables(variables, "conflict_ema")
        self.diagnostic_conflict_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_conflict_sum"
        )
        self.diagnostic_correction_ratio_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_correction_ratio_sum"
        )
        self.diagnostic_update_alignment_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_update_alignment_sum"
        )
        self.diagnostic_opponent_norm_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_opponent_norm_sum"
        )
        self.diagnostic_correction_active_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_correction_active_sum"
        )
        self.diagnostic_count = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_count"
        )

    def _l2_norm(self, tensor) -> Any:
        tensor32 = ops.cast(tensor, "float32")
        return ops.sqrt(ops.sum(ops.square(tensor32)))

    def _projection(self, gradient, momentum):
        denom = ops.sum(ops.square(momentum)) + self.eps
        numer = ops.sum(gradient * momentum)
        scale = numer / denom
        return scale * momentum

    def _cosine_similarity(self, left, right):
        left_norm = self._l2_norm(left)
        right_norm = self._l2_norm(right)
        return ops.sum(left * right) / ((left_norm * right_norm) + self.eps)

    def _opponent_signal(self, gradient, momentum, previous_gradient, gradient_ema):
        if self.opponent_source == "blended":
            blend = ops.cast(self.opponent_blend, gradient.dtype)
            one = ops.cast(1.0, gradient.dtype)
            return (blend * momentum) + ((one - blend) * gradient_ema)
        if self.opponent_source == "previous_gradient":
            return previous_gradient
        if self.opponent_source == "gradient_ema":
            return gradient_ema
        return momentum

    def _conflict_ratio(self, gradient, momentum):
        grad_norm = self._l2_norm(gradient)
        momentum_norm = self._l2_norm(momentum)
        cosine = ops.sum(gradient * momentum) / ((grad_norm * momentum_norm) + self.eps)
        return ops.maximum(ops.cast(0.0, gradient.dtype), -cosine)

    def _adaptive_scale(self, gradient, opponent, conflict_ratio, conflict_ema):
        if not self.adaptive_correction:
            return ops.cast(1.0, gradient.dtype)

        grad_norm = self._l2_norm(gradient)
        opponent_norm = self._l2_norm(opponent)
        reliability = opponent_norm / (grad_norm + opponent_norm + self.eps)
        signal = ops.maximum(conflict_ratio, conflict_ema)
        scale = ops.cast(1.0, gradient.dtype) + reliability + signal
        min_scale = ops.cast(self.adaptive_correction_min_scale, gradient.dtype)
        max_scale = ops.cast(self.adaptive_correction_max_scale, gradient.dtype)
        return ops.clip(scale, min_scale, max_scale)

    def _compute_nce(self, gradient, opponent, step, conflict_ema):
        if self.nce_mode == "off":
            return ops.zeros_like(gradient), ops.cast(0.0, gradient.dtype)

        conflict_ratio = self._conflict_ratio(gradient, opponent)
        if self.nce_mode == "cosine":
            direction = gradient
        else:
            direction = self._projection(gradient, opponent)

        adaptive_scale = self._adaptive_scale(
            gradient,
            opponent,
            conflict_ratio,
            conflict_ema,
        )
        correction = (
            -ops.cast(self.alpha, gradient.dtype)
            * adaptive_scale
            * conflict_ratio
            * direction
        )
        correction_norm = self._l2_norm(correction)
        grad_norm = self._l2_norm(gradient)
        clip_limit = ops.cast(self.nce_clip_ratio, gradient.dtype) * grad_norm
        scale = ops.minimum(
            ops.cast(1.0, gradient.dtype),
            clip_limit / (correction_norm + self.eps),
        )
        correction = correction * scale
        apply_mask = ops.cast(
            step >= ops.cast(self.correction_warmup_steps, step.dtype),
            gradient.dtype,
        )
        if self.conflict_threshold:
            threshold = ops.cast(self.conflict_threshold, gradient.dtype)
            apply_mask = apply_mask * ops.cast(
                conflict_ratio >= threshold,
                gradient.dtype,
            )
        return correction * apply_mask, conflict_ratio

    def _scalar_to_float(self, value) -> float:
        if hasattr(value, "numpy"):
            return float(value.numpy())
        return float(ops.convert_to_numpy(value))

    def _apply_sparsity(self, variable, learning_rate) -> None:
        if self.sparsity_l1:
            shrink = ops.cast(learning_rate * self.sparsity_l1, variable.dtype)
            shrunk = ops.sign(variable) * ops.maximum(
                ops.abs(variable) - shrink,
                ops.cast(0.0, variable.dtype),
            )
            self.assign(variable, shrunk)
        if self.prune_threshold:
            threshold = ops.cast(self.prune_threshold, variable.dtype)
            pruned = ops.where(
                ops.abs(variable) < threshold,
                ops.zeros_like(variable),
                variable,
            )
            self.assign(variable, pruned)

    def update_step(self, gradient, variable, learning_rate) -> None:
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        index = self._get_variable_index(variable)
        momentum = self.momentums[index]
        nce = self.nces[index]
        previous_gradient = self.previous_gradients[index]
        gradient_ema = self.gradient_emas[index]
        conflict_ema = self.conflict_emas[index]
        opponent = self._opponent_signal(
            gradient,
            momentum,
            previous_gradient,
            gradient_ema,
        )
        current_conflict = self._conflict_ratio(gradient, opponent)
        next_conflict_ema = (
            ops.cast(self.adaptive_correction_decay, variable.dtype) * conflict_ema
            + ops.cast(1.0 - self.adaptive_correction_decay, variable.dtype)
            * current_conflict
        )

        correction, conflict_ratio = self._compute_nce(
            gradient,
            opponent,
            self.iterations,
            next_conflict_ema,
        )
        update_direction = gradient + correction
        next_momentum = (
            ops.cast(self.beta, variable.dtype) * momentum
            + ops.cast(1.0 - self.beta, variable.dtype) * update_direction
        )

        self.assign(nce, correction)
        self.assign(momentum, next_momentum)
        next_grad_ema = (
            ops.cast(self.opponent_ema_decay, variable.dtype) * gradient_ema
            + ops.cast(1.0 - self.opponent_ema_decay, variable.dtype) * gradient
        )
        self.assign(previous_gradient, gradient)
        self.assign(gradient_ema, next_grad_ema)
        self.assign(conflict_ema, next_conflict_ema)

        if self.decouple_weight_decay and self.neat_weight_decay:
            decay = ops.cast(self.neat_weight_decay, variable.dtype)
            self.assign_sub(variable, learning_rate * decay * variable)
            self.assign_sub(variable, learning_rate * next_momentum)
        else:
            if self.neat_weight_decay:
                decay_term = ops.cast(self.neat_weight_decay, variable.dtype) * variable
                self.assign_sub(variable, learning_rate * (next_momentum + decay_term))
            else:
                self.assign_sub(variable, learning_rate * next_momentum)
        self._apply_sparsity(variable, learning_rate)

        correction_norm = self._l2_norm(correction)
        grad_norm = self._l2_norm(gradient)
        correction_ratio = correction_norm / (grad_norm + self.eps)
        update_alignment = self._cosine_similarity(gradient, update_direction)
        opponent_norm = self._l2_norm(opponent)
        correction_active = ops.cast(correction_norm > self.eps, "float32")
        self.assign_add(
            self.diagnostic_conflict_sum,
            ops.cast(conflict_ratio, "float32"),
        )
        self.assign_add(
            self.diagnostic_correction_ratio_sum,
            ops.cast(correction_ratio, "float32"),
        )
        self.assign_add(
            self.diagnostic_update_alignment_sum,
            ops.cast(update_alignment, "float32"),
        )
        self.assign_add(
            self.diagnostic_opponent_norm_sum,
            ops.cast(opponent_norm, "float32"),
        )
        self.assign_add(self.diagnostic_correction_active_sum, correction_active)
        self.assign_add(self.diagnostic_count, ops.cast(1.0, "float32"))

    def reset_diagnostics(self) -> None:
        if self.diagnostic_count is None:
            return
        self.assign(self.diagnostic_conflict_sum, 0.0)
        self.assign(self.diagnostic_correction_ratio_sum, 0.0)
        self.assign(self.diagnostic_update_alignment_sum, 0.0)
        self.assign(self.diagnostic_opponent_norm_sum, 0.0)
        self.assign(self.diagnostic_correction_active_sum, 0.0)
        self.assign(self.diagnostic_count, 0.0)

    def diagnostic_snapshot(self) -> dict[str, float]:
        if self.diagnostic_count is None:
            return {}
        count = self._scalar_to_float(self.diagnostic_count)
        if count <= 0.0:
            return {
                "mean_conflict_ratio": 0.0,
                "mean_correction_ratio": 0.0,
                "mean_update_alignment": 1.0,
                "mean_opponent_norm": 0.0,
                "correction_active_fraction": 0.0,
            }
        return {
            "mean_conflict_ratio": (
                self._scalar_to_float(self.diagnostic_conflict_sum) / count
            ),
            "mean_correction_ratio": (
                self._scalar_to_float(self.diagnostic_correction_ratio_sum) / count
            ),
            "mean_update_alignment": (
                self._scalar_to_float(self.diagnostic_update_alignment_sum) / count
            ),
            "mean_opponent_norm": (
                self._scalar_to_float(self.diagnostic_opponent_norm_sum) / count
            ),
            "correction_active_fraction": (
                self._scalar_to_float(self.diagnostic_correction_active_sum) / count
            ),
        }

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "nce_mode": self.nce_mode,
                "eps": self.eps,
                "weight_decay": self.neat_weight_decay,
                "decouple_weight_decay": self.decouple_weight_decay,
                "nce_clip_ratio": self.nce_clip_ratio,
                "sparsity_l1": self.sparsity_l1,
                "prune_threshold": self.prune_threshold,
                "opponent_source": self.opponent_source,
                "opponent_ema_decay": self.opponent_ema_decay,
                "opponent_blend": self.opponent_blend,
                "correction_warmup_steps": self.correction_warmup_steps,
                "conflict_threshold": self.conflict_threshold,
                "adaptive_correction": self.adaptive_correction,
                "adaptive_correction_decay": self.adaptive_correction_decay,
                "adaptive_correction_min_scale": self.adaptive_correction_min_scale,
                "adaptive_correction_max_scale": self.adaptive_correction_max_scale,
            }
        )
        return config
