import numpy as np

from neat_optim import ArrayState, NEATConfig
from neat_optim.engine.reference import neat_group_step_reference


def test_v3_per_parameter_granularity_differs_from_global_group_reduction() -> None:
    params = [np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)]
    grads = [np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)]
    global_states = [ArrayState.zeros_like(param) for param in params]
    local_states = [ArrayState.zeros_like(param) for param in params]
    global_states[0].momentum[:] = -1.0
    global_states[1].momentum[:] = 1.0
    local_states[0].momentum[:] = -1.0
    local_states[1].momentum[:] = 1.0
    base = dict(
        spec_version="nce_spec_v3",
        learning_rate=0.1,
        alpha=0.5,
        beta=0.0,
        native="never",
    )
    global_results = neat_group_step_reference(
        params, grads, global_states, NEATConfig(**base)
    )
    local_results = neat_group_step_reference(
        params,
        grads,
        local_states,
        NEATConfig(conflict_granularity="per_parameter", **base),
    )
    assert global_results[0].metrics.conflict_ratio == 0.0
    assert local_results[0].state.nce[0] < 0.0
    assert local_results[1].state.nce[0] == 0.0
