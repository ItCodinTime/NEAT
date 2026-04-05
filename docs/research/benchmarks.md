# Benchmarks

NEAT now includes a real Keras benchmark on a real classification task in
addition to the smaller sanity checks.

## Benchmark Tasks

- quadratic convergence sanity check
- synthetic alternating-gradient conflict task
- Keras MLP benchmark on `sklearn` digits

## Real Network Comparison

Run date: `2026-04-04`

Task:

- dataset: `sklearn.datasets.load_digits`
- model: MLP with hidden layers `128 -> 64`
- split: `1077` train / `360` validation / `360` test
- epochs: `20`
- seeds: `7`, `11`, `19`
- backend: TensorFlow `2.21.0`
- Keras: `3.13.2`

Comparators:

- SGD with momentum
- Adam
- AdamW
- NEAT Keras optimizer

NEAT was run with a tuned benchmark setting of:

```text
learning_rate = 0.03
alpha         = 0.25
beta          = 0.9
nce_mode      = projection
opponent      = momentum
warmup        = 0
threshold     = 0.0
```

The learning rate was increased from the earlier default-style setting because
the untuned `1e-3` configuration severely under-trained the same network on
this task. The benchmark report below reflects the tuned configuration rather
than that obviously under-fit baseline.

## Results

| Optimizer | Mean Test Accuracy | Std Test Accuracy | Mean Test Loss | Mean Time / Run (s) |
| --- | ---: | ---: | ---: | ---: |
| SGD + momentum | 0.9704 | 0.0065 | 0.1231 | 3.2241 |
| Adam | 0.9685 | 0.0069 | 0.1190 | 3.2667 |
| AdamW | 0.9685 | 0.0069 | 0.1190 | 4.6826 |
| NEAT | 0.9472 | 0.0104 | 0.1991 | 5.0768 |

### NEAT Diagnostics

| Metric | Mean Value |
| --- | ---: |
| Mean conflict ratio | 0.02408 |
| Mean correction ratio | 0.00385 |
| Mean update alignment | 0.99991 |
| Mean opponent norm | 0.13124 |
| Correction active fraction | 1.00000 |

## Interpretation

- NEAT did train the real network successfully and consistently reached
  `94.7%` mean test accuracy on this task.
- On this benchmark, tuned NEAT still trails the standard baselines by roughly
  `2.1` to `2.3` accuracy points.
- SGD with momentum achieved the best mean test accuracy in this run.
- Adam and AdamW achieved the best mean test loss.
- NEAT was the slowest optimizer in the comparison, but still remained within
  the same general wall-clock range.
- The diagnostic view suggests the main issue is not training collapse. The
  mean correction ratio is only `0.00385`, and the mean update alignment is
  `0.99991`, so the NEAT correction barely changes the underlying update on
  this task.

This should be read as an honest early result, not as a claim that NEAT
outperforms standard optimizers today.

## Reproducing

Run:

```bash
python benchmarks/run.py
```

Run the coarse NEAT sweep:

```bash
python benchmarks/sweep_neat.py
```

The default sweep is intentionally coarse so it finishes in a reasonable time.
You can expand the search space by constructing `NEATSweepConfig` directly in
Python and passing it to `run_neat_sweep(...)`.

Machine-readable results for the April 4, 2026 run are stored in:

- `benchmarks/results/keras_mlp_digits_2026-04-04.json`

The benchmark implementation lives in:

- `benchmarks/tasks/keras_mlp.py`
