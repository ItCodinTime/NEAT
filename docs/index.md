# NEAT Documentation

!!! note "Name collision"
    NEAT here means Nash-Equilibrium Adaptive Training. It is unrelated to
    NeuroEvolution of Augmenting Topologies (NEAT; Stanley and Miikkulainen,
    2002).

NEAT is a Keras-first optimizer library built around a Nash-inspired gradient
conflict correction term. The repository is structured so the update rule can
be reasoned about independently of Keras, validated in NumPy, and optionally
accelerated with a native CPU extension.

When the current gradient conflicts with the optimizer's accumulated opponent
signal, NEAT reduces the conflicting component instead of overwriting or
blending it away.

The update rule is kept in a versioned NumPy reference engine, separate from Keras, so it can be studied, tested, and compared independently of any framework. An optional C++ CPU extension drops in behind the same semantics for speed.

---

## Start Here

| You want to… | Go to… |
|---|---|
| Install the package and run your first model | [Quickstart](quickstart.md) |
| Look up a specific class or function | [API Reference](api.md) |
| Understand the math behind the update rule | [Math Spec](research/math-spec.md) |
| Understand why this algorithm was designed this way | [Algorithm Overview](research/algorithm.md) |
| Use per-example gradients in a custom training loop | [Player-Aware Mode](research/player-aware.md) |
| See benchmark results and reproduce them | [Benchmarks](research/benchmarks.md) |
| Contribute code or documentation | [Contributing Guide](contributing/development.md) |

---

## Design Principles

**Explicit math.** The update rule is written as a versioned specification in `docs/research/math-spec.md` and implemented exactly in `engine/reference.py`. The Keras optimizer calls into the same logic rather than reimplementing it.

**Framework independence.** The core algorithm depends only on NumPy. Keras, TensorFlow, and PyTorch are optional. This makes the algorithm testable without a GPU or a large framework stack.

**Small public surface.** The public API is narrow by design. New exports are added only when the existing API cannot cover a clear use case.

**Testability.** The reference engine is pure NumPy and deterministic. Unit tests verify specific numerical values from the math spec's worked example. Integration tests verify that the Keras adapter produces results consistent with the reference engine.

---

## Package Layers

```
docs/research/math-spec.md         Versioned update rule specification
src/neat_optim/engine/reference.py  Canonical NumPy implementation
src/neat_optim/engine/native.py     Optional native CPU extension bridge
src/neat_optim/keras_optimizer.py   Keras optimizer adapter
src/neat_optim/training/            TensorFlow per-player training helpers
src/neat_optim/compaction.py        Dense model pruning and compaction
```

Each layer depends only on the layers below it. Keras-specific code never appears in the reference engine.
