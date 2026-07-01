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
        adaptive_preconditioning: bool = False,
        second_moment_beta: float = 0.999,
        bias_correction: bool = False,
        precondition_nce: bool = True,
        update_mode: str = "momentum",
        adaptive_alpha: bool = False,
        adaptive_alpha_min: float = 0.0,
        adaptive_alpha_max: float = 1.0,
        gradient_noise_decay: float = 0.95,
        gradient_centralization: bool = False,
        nesterov: bool = False,
        lookahead_k: int = 0,
        lookahead_alpha: float = 0.5,
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
            adaptive_preconditioning=adaptive_preconditioning,
            second_moment_beta=second_moment_beta,
            bias_correction=bias_correction,
            precondition_nce=precondition_nce,
            update_mode=update_mode,
            adaptive_alpha=adaptive_alpha,
            adaptive_alpha_min=adaptive_alpha_min,
            adaptive_alpha_max=adaptive_alpha_max,
            gradient_noise_decay=gradient_noise_decay,
            gradient_centralization=gradient_centralization,
            nesterov=nesterov,
            lookahead_k=lookahead_k,
            lookahead_alpha=lookahead_alpha,
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
        self.adaptive_preconditioning = config.adaptive_preconditioning
        self.second_moment_beta = config.second_moment_beta
        self.bias_correction = config.bias_correction
        self.precondition_nce = config.precondition_nce
        self.update_mode = config.update_mode
        self.adaptive_alpha = config.adaptive_alpha
        self.adaptive_alpha_min = config.adaptive_alpha_min
        self.adaptive_alpha_max = config.adaptive_alpha_max
        self.gradient_noise_decay = config.gradient_noise_decay
        self.gradient_centralization = config.gradient_centralization
        self.nesterov = config.nesterov
        self.lookahead_k = config.lookahead_k
        self.lookahead_alpha = config.lookahead_alpha
        self.spec_version = "nce_spec_v2"
        self.momentums = []
        self.nces = []
        self.previous_gradients = []
        self.gradient_emas = []
        self.second_moments = []
        self.slow_weights = []
        self.conflict_emas = []
        self.gradient_noise_emas = []
        self.alignment_emas = []
        self.diagnostic_conflict_sum = None
        self.diagnostic_correction_ratio_sum = None
        self.diagnostic_update_alignment_sum = None
        self.diagnostic_opponent_norm_sum = None
        self.diagnostic_correction_active_sum = None
        self.diagnostic_effective_alpha_sum = None
        self.diagnostic_gradient_noise_sum = None
        self.diagnostic_count = None

    def build(self, variables) -> None:
        """Create one persistent optimizer-state tensor per model variable."""
        if self.built:
            return
        super().build(variables)
        # State arrays mirror ArrayState so Keras and the reference engine can
        # be checked against the same mathematical specification.
        self.momentums = self.add_optimizer_variables(variables, "momentum")
        self.nces = self.add_optimizer_variables(variables, "nce")
        self.previous_gradients = self.add_optimizer_variables(variables, "prev_grad")
        self.gradient_emas = self.add_optimizer_variables(variables, "grad_ema")
        self.second_moments = self.add_optimizer_variables(variables, "second_moment")
        self.slow_weights = self.add_optimizer_variables(variables, "slow_weight")
        self.conflict_emas = self.add_optimizer_variables(variables, "conflict_ema")
        self.gradient_noise_emas = self.add_optimizer_variables(
            variables,
            "grad_noise_ema",
        )
        self.alignment_emas = self.add_optimizer_variables(variables, "alignment_ema")
        # Diagnostics are scalar accumulators rather than parameter-sized
        # buffers, keeping their memory overhead constant as models grow.
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
        self.diagnostic_effective_alpha_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_effective_alpha_sum"
        )
        self.diagnostic_gradient_noise_sum = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_gradient_noise_sum"
        )
        self.diagnostic_count = self.add_variable(
            shape=(), dtype="float32", name="diagnostic_count"
        )

    def _l2_norm(self, tensor) -> Any:
        """Compute norms in float32 to protect mixed-precision stability."""
        tensor32 = ops.cast(tensor, "float32")
        return ops.sqrt(ops.sum(ops.square(tensor32)))

    def _projection(self, gradient, momentum):
        """Project a gradient onto an opponent direction safely."""
        denom = ops.sum(ops.square(momentum)) + self.eps
        numer = ops.sum(gradient * momentum)
        scale = numer / denom
        return scale * momentum

    def _cosine_similarity(self, left, right):
        """Return a numerically stable cosine for arbitrary tensor shapes."""
        left_norm = self._l2_norm(left)
        right_norm = self._l2_norm(right)
        return ops.sum(left * right) / ((left_norm * right_norm) + self.eps)

    def _opponent_signal(self, gradient, momentum, previous_gradient, gradient_ema):
        """Select the configured historical signal used for conflict checks."""
        if self.opponent_source == "blended":
            blend = ops.cast(self.opponent_blend, gradient.dtype)
            one = ops.cast(1.0, gradient.dtype)
            return (blend * momentum) + ((one - blend) * gradient_ema)
        if self.opponent_source == "previous_gradient":
            return previous_gradient
        if self.opponent_source == "gradient_ema":
            return gradient_ema
        return momentum

    def _centralized_gradient(self, gradient):
        """Remove feature-wise means from matrix-like Keras gradients."""
        if not self.gradient_centralization or len(gradient.shape) <= 1:
            return gradient
        axes = tuple(range(len(gradient.shape) - 1))
        return gradient - ops.mean(gradient, axis=axes, keepdims=True)

    def _conflict_ratio(self, gradient, momentum):
        """Map negative cosine alignment to a non-negative conflict score."""
        grad_norm = self._l2_norm(gradient)
        momentum_norm = self._l2_norm(momentum)
        cosine = ops.sum(gradient * momentum) / ((grad_norm * momentum_norm) + self.eps)
        return ops.maximum(ops.cast(0.0, gradient.dtype), -cosine)

    def _adaptive_scale(self, gradient, opponent, conflict_ratio, conflict_ema):
        """Increase correction only when opponent evidence is reliable."""
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

    def _gradient_noise(self, gradient, gradient_ema):
        """Estimate normalized short-term gradient variation."""
        noise_norm = self._l2_norm(gradient - gradient_ema)
        scale = self._l2_norm(gradient) + self._l2_norm(gradient_ema)
        return noise_norm / (scale + ops.cast(self.eps, gradient.dtype))

    def _effective_alpha(self, conflict_ema, gradient_noise_ema, alignment_ema, dtype):
        """Derive the bounded correction coefficient for the current step."""
        if not self.adaptive_alpha:
            return ops.cast(self.alpha, dtype)
        stable_alignment = ops.maximum(ops.cast(0.0, dtype), alignment_ema)
        scale = (
            ops.cast(1.0, dtype)
            + conflict_ema
            + gradient_noise_ema
            - (ops.cast(0.5, dtype) * stable_alignment)
        )
        raw_alpha = ops.cast(self.alpha, dtype) * scale
        return ops.clip(
            raw_alpha,
            ops.cast(self.adaptive_alpha_min, dtype),
            ops.cast(self.adaptive_alpha_max, dtype),
        )

    def _bias_correction(self, beta, step, dtype):
        """Return the finite-step EMA bias-correction denominator."""
        one = ops.cast(1.0, dtype)
        return one - ops.power(ops.cast(beta, dtype), step)

    def _compute_nce(self, gradient, opponent, step, conflict_ema, effective_alpha):
        """Construct and clip the Nash conflict correction tensor."""
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
            -effective_alpha
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
        """Convert a backend scalar for user-facing diagnostic snapshots."""
        if hasattr(value, "numpy"):
            return float(value.numpy())
        return float(ops.convert_to_numpy(value))

    def _apply_sparsity(self, variable, learning_rate) -> None:
        """Apply proximal L1 shrinkage followed by optional hard pruning."""
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
        """Apply one Keras update while mirroring the reference step order."""
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        gradient = self._centralized_gradient(gradient)
        index = self._get_variable_index(variable)
        momentum = self.momentums[index]
        nce = self.nces[index]
        previous_gradient = self.previous_gradients[index]
        gradient_ema = self.gradient_emas[index]
        second_moment = self.second_moments[index]
        slow_weight = self.slow_weights[index]
        conflict_ema = self.conflict_emas[index]
        gradient_noise_ema = self.gradient_noise_emas[index]
        alignment_ema = self.alignment_emas[index]
        step_count = ops.cast(self.iterations + 1, variable.dtype)
        # Second-moment state is updated before conflict correction so both
        # NCE and the final update can share the same preconditioner.
        next_second_moment = (
            ops.cast(self.second_moment_beta, variable.dtype) * second_moment
            + ops.cast(1.0 - self.second_moment_beta, variable.dtype)
            * ops.square(gradient)
        )
        if self.bias_correction:
            second_moment_hat = next_second_moment / self._bias_correction(
                self.second_moment_beta,
                step_count,
                variable.dtype,
            )
        else:
            second_moment_hat = next_second_moment
        preconditioner = ops.sqrt(second_moment_hat) + ops.cast(
            self.eps,
            variable.dtype,
        )
        # Opponent state is historical: it is read before current gradients are
        # written back near the end of this method.
        opponent = self._opponent_signal(
            gradient,
            momentum,
            previous_gradient,
            gradient_ema,
        )
        nce_gradient = gradient
        nce_opponent = opponent
        if self.adaptive_preconditioning and self.precondition_nce:
            nce_gradient = gradient / preconditioner
            nce_opponent = opponent / preconditioner
        current_conflict = self._conflict_ratio(nce_gradient, nce_opponent)
        next_conflict_ema = (
            ops.cast(self.adaptive_correction_decay, variable.dtype) * conflict_ema
            + ops.cast(1.0 - self.adaptive_correction_decay, variable.dtype)
            * current_conflict
        )
        current_alignment = self._cosine_similarity(gradient, previous_gradient)
        current_noise = self._gradient_noise(gradient, gradient_ema)
        next_gradient_noise_ema = (
            ops.cast(self.gradient_noise_decay, variable.dtype) * gradient_noise_ema
            + ops.cast(1.0 - self.gradient_noise_decay, variable.dtype)
            * current_noise
        )
        next_alignment_ema = (
            ops.cast(self.gradient_noise_decay, variable.dtype) * alignment_ema
            + ops.cast(1.0 - self.gradient_noise_decay, variable.dtype)
            * current_alignment
        )
        effective_alpha = self._effective_alpha(
            next_conflict_ema,
            next_gradient_noise_ema,
            next_alignment_ema,
            variable.dtype,
        )

        # NCE may be computed in normalized coordinates, but is always mapped
        # back to gradient coordinates before momentum consumes it.
        correction, conflict_ratio = self._compute_nce(
            nce_gradient,
            nce_opponent,
            self.iterations,
            next_conflict_ema,
            effective_alpha,
        )
        raw_correction = correction
        if self.adaptive_preconditioning and self.precondition_nce:
            raw_correction = correction * preconditioner
        corrected_gradient = gradient + raw_correction
        # Transport the corrected gradient through the selected update family.
        next_momentum = (
            ops.cast(self.beta, variable.dtype) * momentum
            + ops.cast(1.0 - self.beta, variable.dtype) * corrected_gradient
        )
        if self.bias_correction:
            momentum_hat = next_momentum / self._bias_correction(
                self.beta,
                step_count,
                variable.dtype,
            )
        else:
            momentum_hat = next_momentum
        step_update = momentum_hat
        if self.nesterov:
            step_update = (
                ops.cast(self.beta, variable.dtype) * momentum_hat
                + ops.cast(1.0 - self.beta, variable.dtype) * corrected_gradient
            )
        if self.adaptive_preconditioning:
            step_update = momentum_hat / preconditioner
            if self.nesterov:
                step_update = (
                    ops.cast(self.beta, variable.dtype) * momentum_hat
                    + ops.cast(1.0 - self.beta, variable.dtype) * corrected_gradient
                ) / preconditioner
        if self.update_mode == "lion":
            step_update = ops.sign(step_update)

        # Persist history after all reads so the next step—not this one—sees the
        # current gradient as its opponent signal.
        self.assign(nce, raw_correction)
        self.assign(momentum, next_momentum)
        next_grad_ema = (
            ops.cast(self.opponent_ema_decay, variable.dtype) * gradient_ema
            + ops.cast(1.0 - self.opponent_ema_decay, variable.dtype) * gradient
        )
        self.assign(previous_gradient, gradient)
        self.assign(gradient_ema, next_grad_ema)
        self.assign(second_moment, next_second_moment)
        self.assign(conflict_ema, next_conflict_ema)
        self.assign(gradient_noise_ema, next_gradient_noise_ema)
        self.assign(alignment_ema, next_alignment_ema)

        first_step = ops.cast(ops.equal(self.iterations, 0), variable.dtype)
        initialized_slow = (first_step * variable) + (
            (ops.cast(1.0, variable.dtype) - first_step) * slow_weight
        )
        self.assign(slow_weight, initialized_slow)

        # Weight decay precedes proximal sparsity and Lookahead to match the
        # framework-independent reference implementation exactly.
        if self.decouple_weight_decay and self.neat_weight_decay:
            decay = ops.cast(self.neat_weight_decay, variable.dtype)
            self.assign_sub(variable, learning_rate * decay * variable)
            self.assign_sub(variable, learning_rate * step_update)
        else:
            if self.neat_weight_decay:
                decay_term = ops.cast(self.neat_weight_decay, variable.dtype) * variable
                self.assign_sub(variable, learning_rate * (step_update + decay_term))
            else:
                self.assign_sub(variable, learning_rate * step_update)
        self._apply_sparsity(variable, learning_rate)
        if self.lookahead_k:
            sync_step = ops.equal(
                ops.mod(step_count, ops.cast(self.lookahead_k, variable.dtype)),
                ops.cast(0.0, variable.dtype),
            )
            next_slow = initialized_slow + (
                ops.cast(self.lookahead_alpha, variable.dtype)
                * (variable - initialized_slow)
            )
            synced_variable = ops.where(sync_step, next_slow, variable)
            synced_slow = ops.where(sync_step, next_slow, initialized_slow)
            self.assign(variable, synced_variable)
            self.assign(slow_weight, synced_slow)

        # Accumulate backend-native scalars; conversion to Python happens only
        # when diagnostic_snapshot() is explicitly requested.
        correction_norm = self._l2_norm(raw_correction)
        grad_norm = self._l2_norm(gradient)
        correction_ratio = correction_norm / (grad_norm + self.eps)
        update_alignment = self._cosine_similarity(gradient, corrected_gradient)
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
        self.assign_add(
            self.diagnostic_effective_alpha_sum,
            ops.cast(ops.mean(effective_alpha), "float32"),
        )
        self.assign_add(
            self.diagnostic_gradient_noise_sum,
            ops.cast(current_noise, "float32"),
        )
        self.assign_add(self.diagnostic_count, ops.cast(1.0, "float32"))

    def reset_diagnostics(self) -> None:
        """Reset all diagnostic accumulators without touching model state."""
        if self.diagnostic_count is None:
            return
        self.assign(self.diagnostic_conflict_sum, 0.0)
        self.assign(self.diagnostic_correction_ratio_sum, 0.0)
        self.assign(self.diagnostic_update_alignment_sum, 0.0)
        self.assign(self.diagnostic_opponent_norm_sum, 0.0)
        self.assign(self.diagnostic_correction_active_sum, 0.0)
        self.assign(self.diagnostic_effective_alpha_sum, 0.0)
        self.assign(self.diagnostic_gradient_noise_sum, 0.0)
        self.assign(self.diagnostic_count, 0.0)

    def diagnostic_snapshot(self) -> dict[str, float]:
        """Return means of diagnostics collected since the last reset."""
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
                "mean_effective_alpha": 0.0,
                "mean_gradient_noise": 0.0,
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
            "mean_effective_alpha": (
                self._scalar_to_float(self.diagnostic_effective_alpha_sum) / count
            ),
            "mean_gradient_noise": (
                self._scalar_to_float(self.diagnostic_gradient_noise_sum) / count
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
                "adaptive_preconditioning": self.adaptive_preconditioning,
                "second_moment_beta": self.second_moment_beta,
                "bias_correction": self.bias_correction,
                "precondition_nce": self.precondition_nce,
                "update_mode": self.update_mode,
                "adaptive_alpha": self.adaptive_alpha,
                "adaptive_alpha_min": self.adaptive_alpha_min,
                "adaptive_alpha_max": self.adaptive_alpha_max,
                "gradient_noise_decay": self.gradient_noise_decay,
                "gradient_centralization": self.gradient_centralization,
                "nesterov": self.nesterov,
                "lookahead_k": self.lookahead_k,
                "lookahead_alpha": self.lookahead_alpha,
            }
        )
        return config
