"""Optional PyTorch adapter for the NEAT update rule.

PyTorch is intentionally not a core dependency. Import this module only after
installing the ``torch`` extra.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "TorchNEAT requires PyTorch. Install `neat-optim[torch]`."
    ) from exc

from neat_optim.config import NEATConfig


class TorchNEAT(torch.optim.Optimizer):
    """PyTorch implementation of NEAT for research benchmark workloads.

    All algorithm settings are validated by :class:`NEATConfig`. Parameter
    groups may override ``lr`` and ``weight_decay`` in the usual PyTorch way.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        learning_rate: float | None = None,
        diagnostic_interval: int = 1,
        **kwargs: Any,
    ) -> None:
        if diagnostic_interval < 1:
            raise ValueError("diagnostic_interval must be at least 1")
        learning_rate = float(
            kwargs.pop("lr", 1e-3) if learning_rate is None else learning_rate
        )
        config = NEATConfig(learning_rate=learning_rate, native="never", **kwargs)
        self.config = config
        # Diagnostics require several tensor reductions. Sampling them makes
        # accelerator benchmarks cheaper without changing optimizer updates.
        self.diagnostic_interval = diagnostic_interval
        defaults = {"lr": learning_rate, "weight_decay": config.weight_decay}
        super().__init__(params, defaults)
        self._diagnostics: defaultdict[str, float | torch.Tensor] = defaultdict(float)
        self._diagnostic_count = 0

    @staticmethod
    def _norm(value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(value.float())

    def _cosine(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sum(left.float() * right.float()) / (
            self._norm(left) * self._norm(right) + self.config.eps
        )

    def _opponent(self, state: dict[str, Any]) -> torch.Tensor:
        config = self.config
        if config.opponent_source == "previous_gradient":
            return state["previous_gradient"]
        if config.opponent_source == "gradient_ema":
            return state["gradient_ema"]
        if config.opponent_source == "blended":
            return (
                config.opponent_blend * state["momentum"]
                + (1.0 - config.opponent_blend) * state["gradient_ema"]
            )
        return state["momentum"]

    def _record(
        self,
        gradient: torch.Tensor,
        opponent: torch.Tensor,
        correction: torch.Tensor,
        corrected: torch.Tensor,
        conflict: torch.Tensor,
        alpha: torch.Tensor,
        noise: torch.Tensor,
    ) -> None:
        """Accumulate device-side scalars without synchronizing every update."""
        grad_norm = self._norm(gradient)
        correction_norm = self._norm(correction)
        values = {
            "mean_conflict_ratio": conflict,
            "mean_correction_ratio": correction_norm / (grad_norm + self.config.eps),
            "mean_update_alignment": self._cosine(gradient, corrected),
            "mean_opponent_norm": self._norm(opponent),
            "correction_active_fraction": (
                correction_norm > self.config.eps
            ).float(),
            "mean_effective_alpha": alpha,
            "mean_gradient_noise": noise,
        }
        for key, value in values.items():
            self._diagnostics[key] += value
        self._diagnostic_count += 1

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        config = self.config
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", config.weight_decay))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("TorchNEAT does not support sparse gradients")

                # Work directly on the detached gradient because optimizers do
                # not participate in autograd's backward graph.
                gradient = parameter.grad.detach()
                if config.gradient_centralization and gradient.ndim > 1:
                    gradient = gradient - gradient.mean(
                        dim=tuple(range(1, gradient.ndim)), keepdim=True
                    )
                state = self.state[parameter]
                if not state:
                    # PyTorch creates optimizer state lazily so unused model
                    # parameters do not consume momentum-sized buffers.
                    state["step"] = 0
                    for name in (
                        "momentum",
                        "nce",
                        "previous_gradient",
                        "gradient_ema",
                        "second_moment",
                    ):
                        state[name] = torch.zeros_like(parameter)
                    state["slow_param"] = parameter.detach().clone()
                    scalar = torch.zeros((), device=parameter.device)
                    state["conflict_ema"] = scalar.clone()
                    state["gradient_noise_ema"] = scalar.clone()
                    state["alignment_ema"] = scalar.clone()

                step_count = int(state["step"]) + 1

                # Track the diagonal second moment before computing the
                # opponent correction; this mirrors the NumPy reference rule.
                second_moment = state["second_moment"]
                second_moment.mul_(config.second_moment_beta).addcmul_(
                    gradient, gradient, value=1.0 - config.second_moment_beta
                )
                moment_hat = second_moment
                if config.bias_correction:
                    moment_hat = moment_hat / (
                        1.0 - config.second_moment_beta**step_count
                    )
                preconditioner = moment_hat.sqrt().add(config.eps)
                # The opponent is the historical signal against which current
                # gradient conflict is measured.
                opponent = self._opponent(state).clone()
                nce_gradient, nce_opponent = gradient, opponent
                if config.adaptive_preconditioning and config.precondition_nce:
                    nce_gradient = gradient / preconditioner
                    nce_opponent = opponent / preconditioner

                # Negative cosine similarity is the conflict gate. Consistent
                # gradients produce zero correction.
                conflict = torch.clamp(
                    -self._cosine(nce_gradient, nce_opponent), min=0.0
                )
                state["conflict_ema"] = (
                    config.adaptive_correction_decay * state["conflict_ema"]
                    + (1.0 - config.adaptive_correction_decay) * conflict
                )
                alignment = self._cosine(gradient, state["previous_gradient"])
                noise = self._norm(gradient - state["gradient_ema"]) / (
                    self._norm(gradient)
                    + self._norm(state["gradient_ema"])
                    + config.eps
                )
                state["gradient_noise_ema"] = (
                    config.gradient_noise_decay * state["gradient_noise_ema"]
                    + (1.0 - config.gradient_noise_decay) * noise
                )
                state["alignment_ema"] = (
                    config.gradient_noise_decay * state["alignment_ema"]
                    + (1.0 - config.gradient_noise_decay) * alignment
                )
                effective_alpha = torch.as_tensor(
                    config.alpha, device=gradient.device
                )
                if config.adaptive_alpha:
                    scale = (
                        1.0
                        + state["conflict_ema"]
                        + state["gradient_noise_ema"]
                        - 0.5 * torch.clamp(state["alignment_ema"], min=0.0)
                    )
                    effective_alpha = torch.clamp(
                        config.alpha * scale,
                        min=config.adaptive_alpha_min,
                        max=config.adaptive_alpha_max,
                    )

                # NCE removes only the conflicting projected component and is
                # clipped relative to the base gradient for stability.
                correction = torch.zeros_like(gradient)
                if (
                    config.nce_mode != "off"
                    and state["step"] >= config.correction_warmup_steps
                ):
                    if config.nce_mode == "cosine":
                        direction = nce_gradient
                    else:
                        direction = (
                            torch.sum(nce_gradient * nce_opponent)
                            / (torch.sum(nce_opponent.square()) + config.eps)
                        ) * nce_opponent
                    adaptive_scale = torch.ones((), device=gradient.device)
                    if config.adaptive_correction:
                        reliability = self._norm(nce_opponent) / (
                            self._norm(nce_gradient)
                            + self._norm(nce_opponent)
                            + config.eps
                        )
                        adaptive_scale = torch.clamp(
                            1.0
                            + reliability
                            + torch.maximum(conflict, state["conflict_ema"]),
                            min=config.adaptive_correction_min_scale,
                            max=config.adaptive_correction_max_scale,
                        )
                    correction = (
                        -effective_alpha * adaptive_scale * conflict * direction
                    )
                    if config.conflict_threshold:
                        correction.mul_(
                            (conflict >= config.conflict_threshold).to(
                                correction.dtype
                            )
                        )
                    limit = config.nce_clip_ratio * self._norm(nce_gradient)
                    correction.mul_(
                        torch.minimum(
                            torch.ones((), device=gradient.device),
                            limit / (self._norm(correction) + config.eps),
                        )
                    )
                if config.adaptive_preconditioning and config.precondition_nce:
                    correction = correction * preconditioner
                corrected = gradient + correction

                # Feed the corrected gradient through momentum and optional
                # Adam-style diagonal preconditioning.
                momentum = state["momentum"]
                momentum.mul_(config.beta).add_(corrected, alpha=1.0 - config.beta)
                update = momentum
                if config.bias_correction:
                    update = update / (1.0 - config.beta**step_count)
                if config.nesterov:
                    update = config.beta * update + (1.0 - config.beta) * corrected
                if config.adaptive_preconditioning:
                    update = update / preconditioner
                if config.update_mode == "lion":
                    update = update.sign()

                # Apply the update, then proximal sparsity and Lookahead in the
                # same order as the framework-independent reference engine.
                if config.decouple_weight_decay and weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(update, alpha=-lr)
                else:
                    parameter.add_(update + weight_decay * parameter, alpha=-lr)
                if config.sparsity_l1:
                    parameter.copy_(
                        parameter.sign()
                        * torch.clamp(parameter.abs() - lr * config.sparsity_l1, min=0)
                    )
                if config.prune_threshold:
                    parameter.masked_fill_(parameter.abs() < config.prune_threshold, 0)
                if config.lookahead_k and step_count % config.lookahead_k == 0:
                    state["slow_param"].add_(
                        parameter - state["slow_param"], alpha=config.lookahead_alpha
                    )
                    parameter.copy_(state["slow_param"])

                # Persist history only after the update so opponent signals
                # always refer to information from earlier steps.
                state["nce"].copy_(correction)
                state["previous_gradient"].copy_(gradient)
                state["gradient_ema"].mul_(config.opponent_ema_decay).add_(
                    gradient, alpha=1.0 - config.opponent_ema_decay
                )
                state["step"] = step_count
                if (step_count - 1) % self.diagnostic_interval == 0:
                    self._record(
                        gradient,
                        opponent,
                        correction,
                        corrected,
                        conflict,
                        effective_alpha,
                        noise,
                    )
        return loss

    def reset_diagnostics(self) -> None:
        """Clear accumulated optimizer diagnostics."""
        self._diagnostics.clear()
        self._diagnostic_count = 0

    def diagnostic_snapshot(self) -> dict[str, float]:
        """Return mean diagnostics accumulated since the last reset."""
        if not self._diagnostic_count:
            return {}
        return {
            key: float(value / self._diagnostic_count)
            for key, value in self._diagnostics.items()
        }
