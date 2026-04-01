# Benchmarks

The benchmark suite starts with small, reproducible tasks:

- quadratic convergence sanity check
- synthetic alternating-gradient conflict task
- optional Keras MLP benchmark

Comparators for the first release line:

- SGD with momentum
- NEAT reference engine

Adam and AdamW should be added once the Keras integration benchmark is wired to
the runtime backend in CI.
