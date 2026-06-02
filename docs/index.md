# NEAT

NEAT is a Keras-first optimizer library built around a Nash-inspired gradient
conflict correction term. The repository is structured so the update rule can
be reasoned about independently of Keras, validated in NumPy, and optionally
accelerated with a native CPU extension.

The package now exposes two distinct usage styles:

- standard NEAT for aggregated batch gradients through the Keras optimizer API
- player-aware NEAT for explicit per-example or per-task gradients in a custom
  TensorFlow training loop

The repository also includes benchmark diagnostics and a NEAT sweep harness so
optimizer changes can be measured instead of argued from theory alone.

## Design Goals

- Keep the public API small and production-friendly.
- Keep the optimizer math explicit and versioned.
- Keep the core testable outside any single deep-learning framework.
- Keep native acceleration behind the same semantics as the reference engine.

## Package Layers

1. `docs/research/math-spec.md`
   The versioned mathematical contract for the update rule.
2. `src/neat_optim/engine/reference.py`
   The canonical NumPy implementation used for correctness validation.
3. `src/neat_optim/engine/native.py`
   Optional bridge to the native CPU kernel.
4. `src/neat_optim/keras_optimizer.py`
   Keras optimizer adapter for model training.
5. `src/neat_optim/engine/multiplayer.py` and `src/neat_optim/training/`
   Explicit per-player stepping and TensorFlow helpers.
6. `tests/`, `examples/`, and `benchmarks/`
   Product-layer validation, usage, and measurement.

## Start Here

- For installation and first usage, read `Quickstart`.
- For the public API surface, read `API`.
- For algorithm details, read `Research`.
- For local development and release workflow, read `Contributing`.
