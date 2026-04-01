from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def sample_param() -> np.ndarray:
    return np.array([1.0, -2.0, 0.5], dtype=np.float32)


@pytest.fixture()
def sample_grad() -> np.ndarray:
    return np.array([0.5, -0.25, 0.75], dtype=np.float32)
