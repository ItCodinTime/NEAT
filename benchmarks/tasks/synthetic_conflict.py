from __future__ import annotations

import numpy as np


def make_conflicting_gradient(step: int, size: int = 128) -> np.ndarray:
    base = np.ones((size,), dtype=np.float32)
    return base if step % 2 == 0 else -base
