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
        self.spec_version = "nce_spec_v1"
        self.momentums = []
        self.nces = []

    def build(self, variables) -> None:
        if self.built:
            return
        super().build(variables)
        self.momentums, self.nces = self.add_optimizer_variables(
            variables, ["momentum", "nce"]
        )

    def _l2_norm(self, tensor) -> Any:
        tensor32 = ops.cast(tensor, "float32")
        return ops.sqrt(ops.sum(ops.square(tensor32)) + self.eps)

    def _projection(self, gradient, momentum):
        denom = ops.sum(ops.square(momentum)) + self.eps
        numer = ops.sum(gradient * momentum)
        scale = numer / denom
        return scale * momentum

    def _conflict_ratio(self, gradient, momentum):
        grad_norm = self._l2_norm(gradient)
        momentum_norm = self._l2_norm(momentum)
        cosine = ops.sum(gradient * momentum) / ((grad_norm * momentum_norm) + self.eps)
        return ops.maximum(ops.cast(0.0, gradient.dtype), -cosine)

    def _compute_nce(self, gradient, momentum):
        if self.nce_mode == "off":
            return ops.zeros_like(gradient), ops.cast(0.0, gradient.dtype)

        conflict_ratio = self._conflict_ratio(gradient, momentum)
        if self.nce_mode == "cosine":
            direction = gradient
        else:
            direction = self._projection(gradient, momentum)

        correction = -ops.cast(self.alpha, gradient.dtype) * conflict_ratio * direction
        correction_norm = self._l2_norm(correction)
        grad_norm = self._l2_norm(gradient)
        clip_limit = ops.cast(self.nce_clip_ratio, gradient.dtype) * grad_norm
        scale = ops.minimum(
            ops.cast(1.0, gradient.dtype),
            clip_limit / (correction_norm + self.eps),
        )
        return correction * scale, conflict_ratio

    def update_step(self, gradient, variable, learning_rate) -> None:
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        index = self._get_variable_index(variable)
        momentum = self.momentums[index]
        nce = self.nces[index]

        correction, _ = self._compute_nce(gradient, momentum)
        update_direction = gradient + correction
        next_momentum = (
            ops.cast(self.beta, variable.dtype) * momentum
            + ops.cast(1.0 - self.beta, variable.dtype) * update_direction
        )

        self.assign(nce, correction)
        self.assign(momentum, next_momentum)

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
            }
        )
        return config
