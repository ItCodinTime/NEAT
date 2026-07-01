# Benchmark Plan

This page defines the benchmark bar for claims beyond the small local tasks
already checked into this repository.

## Minimum Comparator Set

Use the same model, data split, precision policy, batch size, gradient clipping,
and wall-clock budget for every optimizer.

- AdamW
- SGD with momentum where it is a meaningful baseline
- Lion
- Sophia or a documented diagonal-Hessian approximation baseline
- Adafactor for memory-sensitive transformer runs
- Muon where the model architecture and parameter shapes make it applicable
- NEAT default
- NEAT adaptive stack:
  `adaptive_alpha=True`, `adaptive_preconditioning=True`,
  `opponent_source="previous_gradient"`
- NEAT sign stack:
  `update_mode="lion"`, `adaptive_alpha=True`

## Required Reporting

Every benchmark report should include:

- exact command line
- git commit
- hardware
- backend versions
- seeds
- full optimizer config
- train, validation, and test metrics
- wall-clock time
- peak memory when available
- NEAT diagnostics:
  `mean_conflict_ratio`, `mean_correction_ratio`,
  `mean_update_alignment`, `correction_active_fraction`,
  `mean_effective_alpha`, and `mean_gradient_noise`
- raw logs in TensorBoard or Weights & Biases

## Target Suites

The executable implementations and exact starter commands live in
[`benchmarks/torch_suite/README.md`](../../benchmarks/torch_suite/README.md).

### Vision

- ResNet-50 on ImageNet-100 as the minimum credible ImageNet-style target
- ImageNet-1K only when enough GPU budget is available
- Vision Transformer on CIFAR-100 or ImageNet-100

### Language Model Fine-Tuning

- LoRA or full fine-tuning must be controlled outside the optimizer so every
  optimizer sees the same trainable parameter set.
- Candidate tasks: Alpaca-style instruction tuning and ShareGPT-style chat
  tuning.
- Candidate models: Llama-family or Mistral-family checkpoints that the runner
  is licensed to use.

### Diffusion

- Use an existing training recipe and change only the optimizer.
- Report FID or another standard sample-quality metric, not only training loss.

### Reinforcement Learning

- Use stable-baselines-style task definitions and report multi-seed returns.
- Optimizer comparisons should include variance, not only best seed.

## Claim Policy

Do not describe NEAT as state of the art from a single small task. A result can
be promoted in the README only when it has reproducible scripts, logs, multiple
seeds, and a clear comparator table.
