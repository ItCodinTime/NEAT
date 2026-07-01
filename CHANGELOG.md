# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic
Versioning once `1.0.0` is reached. Before that, breaking changes may occur in
minor releases.

## [Unreleased]

### Added

- Optional PyTorch `TorchNEAT` adapter and modern vision, LLM, RL, diffusion,
  and large-batch benchmark runners
- Advanced cross-framework parity tests for adaptive, Lion, Nesterov,
  Lookahead, regularization, sparsity, and pruning modes
- Algorithm-phase comments and API documentation across optimizer engines and
  benchmark infrastructure

### Changed

- PyTorch benchmark diagnostics are sampled every ten steps to reduce
  accelerator synchronization overhead without changing updates

## [0.1.0.dev0] - 2026-04-01

### Added

- Initial repository scaffold
- NumPy reference implementation of the NEAT optimizer
- Keras optimizer integration scaffold
- Optional native CPU core scaffold
- Unit, regression, and integration test layout
- Benchmarks, examples, docs, and CI workflows
