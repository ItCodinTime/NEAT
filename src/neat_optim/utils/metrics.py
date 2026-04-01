from __future__ import annotations

import numpy as np


def l2_norm(array: np.ndarray, eps: float = 0.0) -> float:
    return float(np.sqrt(np.sum(np.square(array, dtype=np.float64)) + eps))
