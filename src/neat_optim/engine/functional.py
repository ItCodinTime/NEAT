"""Framework-agnostic public stepping API for NEAT."""

from __future__ import annotations

import numpy as np

from neat_optim.config import NEATConfig
from neat_optim.engine.common import (
    active_fraction,
    as_float32,
    cosine_similarity,
    safe_ratio,
)
from neat_optim.engine.native import load_native_core
from neat_optim.engine.reference import neat_step_reference
from neat_optim.exceptions import NativeCoreUnavailableError
from neat_optim.state import ArrayState, StepMetrics, StepResult
from neat_optim.utils.metrics import l2_norm


def neat_step(
    param: np.ndarray,
    grad: np.ndarray,
    state: ArrayState,
    config: NEATConfig,
) -> StepResult:
    """Apply one optimizer step, using the native core when available.

    This function is the recommended public entrypoint when you want a
    framework-independent NEAT step over NumPy arrays while still allowing the
    optional native core to accelerate execution.
    """

    if (
        config.native != "never"
        and config.sparsity_l1 == 0.0
        and config.prune_threshold == 0.0
        and config.opponent_source == "momentum"
        and config.opponent_blend == 0.5
        and config.correction_warmup_steps == 0
        and config.conflict_threshold == 0.0
        and not config.adaptive_correction
        and config.adaptive_correction_decay == 0.9
        and config.adaptive_correction_min_scale == 1.0
        and config.adaptive_correction_max_scale == 3.0
        and not config.adaptive_preconditioning
        and config.second_moment_beta == 0.999
        and not config.bias_correction
        and config.precondition_nce
    ):
        try:
            pre_momentum = as_float32(state.momentum).copy()
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
            state.previous_gradient = as_float32(grad).copy()
            previous_ema = (
                np.zeros_like(grad, dtype=np.float32)
                if state.gradient_ema is None
                else as_float32(state.gradient_ema)
            )
            state.gradient_ema = (
                (config.opponent_ema_decay * previous_ema)
                + ((1.0 - config.opponent_ema_decay) * as_float32(grad))
            ).astype(np.float32, copy=False)
            if state.second_moment is None:
                state.second_moment = np.zeros_like(grad, dtype=np.float32)
            state.step += 1
            grad32 = as_float32(grad)
            nce32 = as_float32(state.nce)
            grad_norm = float(native_metrics["grad_norm"])
            nce_norm = float(native_metrics["nce_norm"])
            metrics = StepMetrics(
                grad_norm=grad_norm,
                update_norm=float(native_metrics["update_norm"]),
                nce_norm=nce_norm,
                conflict_ratio=float(native_metrics["conflict_ratio"]),
                active_fraction=active_fraction(as_float32(param)),
                opponent_norm=l2_norm(pre_momentum),
                correction_ratio=safe_ratio(nce_norm, grad_norm, config.eps),
                update_alignment=cosine_similarity(grad32, grad32 + nce32, config.eps),
                correction_active=1.0 if nce_norm > config.eps else 0.0,
            )
            return StepResult(param=param, state=state, metrics=metrics)
        except NativeCoreUnavailableError:
            if config.native == "force":
                raise

    return neat_step_reference(param, grad, state, config)
