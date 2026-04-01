from __future__ import annotations

import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.native import load_native_core
from neat_optim.engine.reference import neat_step_reference
from neat_optim.exceptions import NativeCoreUnavailableError
from neat_optim.state import ArrayState, StepMetrics, StepResult


def neat_step(
    param: np.ndarray,
    grad: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> StepResult:
    """Apply one optimizer step, using the native core when available."""

    if config.native != "never":
        try:
            native = load_native_core()
            native_metrics = native.cpu_step_inplace(
                param,
                grad,
                state.momentum,
                state.nce,
                config.learning_rate,
                config.alpha,
                config.beta,
                config.eps,
                config.weight_decay,
                config.nce_clip_ratio,
                config.nce_mode,
                config.decouple_weight_decay,
            )
            state.step += 1
            metrics = StepMetrics(
                grad_norm=float(native_metrics["grad_norm"]),
                update_norm=float(native_metrics["update_norm"]),
                nce_norm=float(native_metrics["nce_norm"]),
                conflict_ratio=float(native_metrics["conflict_ratio"]),
            )
            return StepResult(param=param, state=state, metrics=metrics)
        except NativeCoreUnavailableError:
            if config.native == "force":
                raise

    return neat_step_reference(param, grad, state, config)
