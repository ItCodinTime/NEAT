# Benchmark Configs

This directory is reserved for reproducible benchmark presets and generated
run manifests.

Recommended NEAT presets:

```text
neat_default:
  learning_rate: tune per task
  alpha: 0.25
  beta: 0.9
  nce_mode: projection
  opponent_source: momentum

neat_adaptive:
  learning_rate: tune per task
  alpha: 0.25
  beta: 0.9
  nce_mode: projection
  opponent_source: previous_gradient
  adaptive_alpha: true
  adaptive_preconditioning: true
  bias_correction: true

neat_lion:
  learning_rate: tune per task
  alpha: 0.2
  beta: 0.9
  update_mode: lion
  adaptive_alpha: true
  gradient_centralization: true
```

Before publishing results, record the full config, commit, hardware, seeds,
backend versions, raw logs, and NEAT diagnostic snapshot keys listed in
`docs/research/benchmark-plan.md`.
