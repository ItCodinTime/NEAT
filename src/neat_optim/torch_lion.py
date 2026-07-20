"""Minimal PyTorch implementation of the published Lion update transport."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "TorchLion requires PyTorch. Install `neat-optim[torch]`."
    ) from exc


class TorchLion(torch.optim.Optimizer):
    """Lion transport used to isolate NEAT correction from optimizer effects."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(
            params, {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            beta_1, beta_2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("TorchLion does not support sparse gradients")
                gradient = parameter.grad.detach()
                state = self.state[parameter]
                if not state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                if weight_decay:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                update = beta_1 * momentum + (1.0 - beta_1) * gradient
                parameter.add_(update.sign(), alpha=-learning_rate)
                momentum.mul_(beta_2).add_(gradient, alpha=1.0 - beta_2)
        return loss
