# Quickstart

## Install

```bash
pip install "neat-optim[keras]" tensorflow
```

## Use With Keras

```python
from neat_optim import NEAT

optimizer = NEAT(
    learning_rate=1e-3,
    alpha=0.25,
    beta=0.9,
    nce_mode="projection",
)
```

## Use The Reference Engine

```python
import numpy as np
from neat_optim import NEATConfig
from neat_optim.engine.functional import neat_step
from neat_optim.state import ArrayState

param = np.array([1.0, -2.0], dtype=np.float32)
grad = np.array([0.5, -0.25], dtype=np.float32)
state = ArrayState.zeros_like(param)
result = neat_step(param, grad, state, NEATConfig())
```
