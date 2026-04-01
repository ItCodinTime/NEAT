# Architecture

NEAT is built in layers:

1. Versioned math spec
2. NumPy reference engine
3. Optional native CPU acceleration for the reference engine
4. Keras optimizer integration
5. Product layer: tests, docs, benchmarks, packaging, and CI

The core rule is that novel update logic lives in the engine, not in the
framework adapter.
