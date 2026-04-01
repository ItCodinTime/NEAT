# NEAT

NEAT is a Keras-first optimizer library built around Nash-equilibrium-inspired
gradient conflict correction.

The project is organized around a simple rule:

- keep the public API small
- make the math explicit
- keep the core testable outside any one framework
- isolate performance-sensitive code behind a narrow native boundary
