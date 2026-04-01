import numpy as np
import pytest

from neat_optim.config import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.engine.native import load_native_core
from neat_optim.engine.reference import neat_step_reference
from neat_optim.exceptions import NativeCoreUnavailableError
from neat_optim.state import ArrayState


def test_native_matches_reference_when_available() -> None:
    try:
        load_native_core()
    except NativeCoreUnavailableError:
        pytest.skip("native core not installed")

    param = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    grad = np.array([0.5, -0.25, 0.75], dtype=np.float32)
    base_state = ArrayState(
        momentum=np.array([0.2, -0.1, 0.0], dtype=np.float32),
        nce=np.zeros(3, dtype=np.float32),
        step=4,
    )
    config = NEATConfig(
        learning_rate=1e-2,
        alpha=0.25,
        beta=0.9,
        weight_decay=1e-3,
        nce_mode="projection",
    )

    native_state = ArrayState(
        momentum=base_state.momentum.copy(),
        nce=base_state.nce.copy(),
        step=base_state.step,
    )
    reference_state = ArrayState(
        momentum=base_state.momentum.copy(),
        nce=base_state.nce.copy(),
        step=base_state.step,
    )
    native_param = param.copy()
    reference_param = param.copy()

    native_result = neat_step(native_param, grad, native_state, config)
    reference_result = neat_step_reference(
        reference_param, grad, reference_state, config
    )

    np.testing.assert_allclose(native_result.param, reference_result.param, atol=1e-6)
    np.testing.assert_allclose(
        native_result.state.momentum, reference_result.state.momentum, atol=1e-6
    )
    np.testing.assert_allclose(
        native_result.state.nce, reference_result.state.nce, atol=1e-6
    )
    assert native_result.metrics.conflict_ratio == pytest.approx(
        reference_result.metrics.conflict_ratio, abs=1e-6
    )
