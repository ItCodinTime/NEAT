# Benchmark run status - 2026-07-28

This note records the benchmark and documentation alignment pass started on
July 28, 2026. It distinguishes completed evidence from planned server work so
future claims cannot blur smoke tests, validation gates, and paper-grade runs.

## Local validation completed

Commands were run from repository revision `9a42fa8` with a clean git tree.
Python used for this pass was `3.12.13` inside `.venv`.

| Check | Command | Result |
|---|---|---|
| Lint | `.venv/bin/ruff check .` | Passed |
| Unit tests | `.venv/bin/pytest tests/unit -q` | `100 passed` |
| Full local tests | `.venv/bin/pytest -q` | `109 passed, 3 skipped` |
| Docs | `.venv/bin/mkdocs build --strict` | Passed |
| Cheap benchmark smoke | `.venv/bin/python -m benchmarks.torch_suite.compare_all --quick --output-dir benchmark-runs/compare-all-quick-2026-07-28` | Passed |

The full local test skips were TensorFlow-dependent integration tests only:

- `tests/integration/test_compaction.py`
- `tests/integration/test_keras_optimizer.py`
- `tests/integration/test_tensorflow_players.py`

The cheap benchmark smoke wrote `40` result manifests under
`benchmark-runs/compare-all-quick-2026-07-28`: five seeds across
`optimizer-smoke` and `multitask-synthetic`. This is machinery evidence only.
It is not a performance benchmark and should not be cited as optimizer quality
evidence.

## Evidence integrity audit

The repository already keeps positive, null, and negative results separated:

- Keras digits remains a negative result for NEAT versus SGD/Adam.
- Short MNIST/Fashion-MNIST CNN is documented as limited transfer evidence.
- CIFAR-10 ResNet-18 MPS is documented as parity, not superiority.
- STS v1, v2 promotion, and conflict-severity runs are documented as negative
  or non-promoting evidence.

An overclaim scan for terms such as `state of the art`, `superiority`,
`outperform`, `breakthrough`, `revolutionary`, and `vibe` found cautious claim
boundaries plus the AI assistance log. The AI assistance log is retained as
provenance. It should not be removed to make the project look less
AI-assisted; rigor comes from reproducible manifests, tests, and conservative
claims.

## Vultr cloud status

After API credentials were supplied for this pass, the account was inspected and
had no existing active instances. GPU provisioning for
`vcg-a16-3c-32g-8vram` in `ord` was rejected by the Vultr API because Cloud GPU
access requires a support request. Three parallel `voc-g-16c-64gb-320s-amd`
CPU workers were also rejected by the account monthly fee limit.

One CPU worker was launched instead:

| Field | Value |
|---|---|
| Instance | `45c7f5e6-f90d-426d-af8d-355da9cd5be0` |
| Plan | `vhf-8c-32gb` |
| Region | `ord` |
| Public IP | `64.177.124.19` |
| Live catalog price | approximately USD `0.286` per hour |
| Run directory | `benchmark-runs/vultr-cpu-2026-07-28-r3` |

The worker checked out repository revision `9a42fa8` with `dirty=0`. The first
remote PyTorch install pulled CUDA/Triton wheels and segfaulted during the full
test suite. That was corrected by replacing PyTorch with CPU-only wheels; the
subsequent run passed `ruff`, full `pytest`, docs, and the dataset-free quick
benchmark on the clean tree.

The remote CIFAR download source was severely throttled from the worker, so
CIFAR-backed vision and large-batch runs are blocked by dataset transfer rather
than by optimizer code or compute. MNIST data was synchronized to the worker so
the diffusion queue could proceed.

Final observed cloud queue:

| Stage | Status |
|---|---|
| `ruff` | Passed |
| `pytest-full` | Passed |
| `docs` | Passed |
| `quick-core` | Passed, `40` manifests copied into the current run directory |
| `backend-smoke-fixed` | Passed across TensorFlow, JAX, and Torch Keras backends |
| `optimizer-smoke-full` | Passed |
| `multitask-full` | Passed |
| `diffusion-quick` | Passed, `20` manifests |
| `rl-quick` | Timed out at 4 hours with `18` manifests |
| `language-prelim` | Failed fast before training due a Hugging Face Hub URI compatibility issue |
| `vision` / `large-batch` | Blocked by CIFAR dataset download throttling |

A local `screen` session named `neatwatch` ran `scripts/vultr_watchdog.py`.
It polled the worker, synchronized the ignored artifact directory locally, and
deleted the instance when the remote run wrote `DONE`. The same watchdog would
also have deleted the instance if the 18-hour timeout was hit or pending account
charges reached USD `150`. It synchronized the artifacts and deleted instance
`45c7f5e6-f90d-426d-af8d-355da9cd5be0` at `2026-07-29T04:47:44Z`. Last
observed pending account charges were USD `17.50`.

The synchronized run directory is
`benchmark-runs/vultr-cpu-2026-07-28-r3` and contains `118` result manifests:

| Directory | Result manifests |
|---|---:|
| `compare-all-quick` | 40 |
| `optimizer-smoke-full` | 20 |
| `multitask-full` | 20 |
| `category-diffusion-quick` | 20 |
| `category-rl-quick` | 18 |

Generated aggregate artifacts:

- `benchmark-runs/vultr-cpu-2026-07-28-r3/report-torch-suite/summary.csv`
- `benchmark-runs/vultr-cpu-2026-07-28-r3/report-torch-suite/convergence.png`
- `benchmark-runs/vultr-cpu-2026-07-28-r3/report-completed-nonpaper/summary.md`
- `benchmark-runs/vultr-cpu-2026-07-28-r3/report-completed-nonpaper/aggregate.json`

Selected cloud metrics, all preliminary until reviewed in the raw manifests:

| Task | Optimizer | Seeds | Metric | Mean |
|---|---|---:|---|---:|
| Diffusion MNIST | Lion | 5 | validation noise MSE, lower is better | `0.03766` |
| Diffusion MNIST | NEAT | 5 | validation noise MSE, lower is better | `0.04497` |
| Diffusion MNIST | AdamW | 5 | validation noise MSE, lower is better | `0.04500` |
| Diffusion MNIST | SGD | 5 | validation noise MSE, lower is better | `0.18208` |
| HalfCheetah-v5 quick | AdamW | 5 | mean episode reward, higher is better | `1873.37` |
| HalfCheetah-v5 quick | NEAT | 5 | mean episode reward, higher is better | `1333.62` |
| HalfCheetah-v5 quick | Lion | 4 | mean episode reward, higher is better | `2584.17` |
| HalfCheetah-v5 quick | SGD | 4 | mean episode reward, higher is better | `981.43` |

The language failure was caused by the benchmark environment resolving
`huggingface_hub 1.25.1`, where `datasets.load_dataset("glue", "sst2")` raised
an invalid `hf://datasets/glue@...` URI error. The benchmark dependency set now
pins `huggingface-hub>=0.34,<1.0`; a dry-run resolver check selected
`huggingface_hub 0.36.2` alongside current `datasets`, `transformers`, `peft`,
and `accelerate`.

## Remaining benchmark work

Remaining work is:

1. Preserve the synchronized cloud artifacts and generate aggregate reports for
   any completed non-preliminary categories. This was completed for the current
   run directory, but the reports should be reviewed before promoting claims.
2. Retry CIFAR-backed vision and large-batch jobs only after either a local
   dataset mirror is available or Vultr/GPU dataset download bandwidth is
   healthy.
3. Run CUDA viability only after Vultr Cloud GPU access is approved.
4. Run the full GPU-backed suite only after the viability check passes and cost
   projection is still within the approved budget.

## Follow-up - 2026-07-29

The local `data/cifar-10-python.tar.gz` file was found to be truncated. It was
preserved as `data/cifar-10-python.tar.gz.corrupt-20260729` and replaced with a
checksum-valid mirror copy:

| Artifact | Checksum |
|---|---|
| `data/cifar-10-python.tar.gz` | MD5 `c58f30108f718f92721af3b95e74349a` |
| `data/cifar-10-python.tar.gz` | SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |

TorchVision now loads local CIFAR-10 as `50,000` train and `10,000` validation
examples without network access. This removes the immediate CIFAR data blocker.

Preliminary dirty-tree local CPU viability checks were run to verify the
benchmark path. These are plumbing and rough-signal runs only; they are
single-seed, one-epoch runs and must not be promoted as final optimizer claims.

| Run directory | Batch | Optimizer | Top-1 accuracy | Wall time |
|---|---:|---|---:|---:|
| `benchmark-runs/cifar10-local-viability-2026-07-29` | 128 | AdamW | `0.5470` | `227.82s` |
| `benchmark-runs/cifar10-local-viability-2026-07-29` | 128 | NEAT | `0.5439` | `260.33s` |
| `benchmark-runs/large-batch-local-viability-2026-07-29` | 256 | AdamW | `0.5395` | `222.58s` |
| `benchmark-runs/large-batch-local-viability-2026-07-29` | 256 | NEAT | `0.5511` | `262.99s` |

The Hugging Face language benchmark dependencies were installed with the new
`huggingface-hub>=0.34,<1.0` constraint. A tiny SST-2 smoke using
`hf-internal-testing/tiny-random-bert` completed and wrote
`benchmark-runs/language-smoke-2026-07-29`. It is only a dependency and runner
smoke, not a language-model quality benchmark.

Remaining true blockers after this follow-up:

- Vultr Cloud GPU account approval is still required for CUDA viability. A live
  API recheck on `2026-07-29` found zero running instances, USD `17.79` pending
  charges, and the `vcg-a16-3c-32g-8vram` create request was still rejected
  with the support-request requirement.
- Full CIFAR and large-batch sweeps should use the verified local CIFAR mirror,
  preferably on approved GPU hardware.
- The real language benchmark still needs an appropriately sized GPU worker and
  a reviewed model choice; the tiny local smoke only proves the runner path.

## Follow-up Vultr run - 2026-07-29

After the user approved using additional Vultr capacity, Cloud GPU creation was
retried for `vcg-a16-3c-32g-8vram` in `ord` and was still rejected by the Vultr
API with the support-request requirement. A second parallel CPU worker was also
rejected by the account monthly fee limit, so the remaining CPU-runnable work
was queued on one worker:

| Field | Value |
|---|---|
| Instance | `f64bc78a-929c-4fd2-96af-00ec5572563d` |
| Plan | `vhf-8c-32gb` |
| Region | `ord` |
| Public IP | `64.177.120.21` |
| Run directory | `benchmark-runs/vultr-remaining-cpu-2026-07-29` |

The worker was bootstrapped with CPU-only PyTorch, TensorFlow CPU, JAX CPU, the
benchmark extras, and the checksum-valid local CIFAR-10 archive. Remote sanity
checks passed before the benchmark queue started: `ruff check .`, full
`pytest -q`, `mkdocs build --strict`, and a CIFAR load check. The initial
`cifar-data-check` entry failed because of a shell quoting mistake in the remote
one-liner, not because the dataset was bad; the corrected check passed.

The local watchdog synchronized artifacts and deleted instance
`f64bc78a-929c-4fd2-96af-00ec5572563d` after the remote queue wrote `DONE` at
`2026-07-29T20:24:58Z`. Last observed pending account charges during the run
were below USD `19`.

Completed follow-up queue:

| Stage | Status |
|---|---|
| `rl-missing-lion-seed29` | Passed |
| `rl-missing-sgd-seed29` | Passed |
| `language-smoke-vultr` | Passed |
| `vision-cifar10-quick` | Passed, `20` manifests |
| `large-batch-cifar10-quick` | Passed, `24` manifests |
| `language-tinyllama-prelim` | Passed, `4` manifests |

The synchronized run directory contains `51` result manifests plus generated
report artifacts under `report-torch-suite/`.

Selected follow-up metrics, all preliminary CPU evidence:

| Task | Optimizer | Seeds | Metric | Mean |
|---|---|---:|---|---:|
| CIFAR-10 ResNet-18, 3 epochs | Lion | 5 | top-1 accuracy, higher is better | `0.7035` |
| CIFAR-10 ResNet-18, 3 epochs | NEAT | 5 | top-1 accuracy, higher is better | `0.6857` |
| CIFAR-10 ResNet-18, 3 epochs | AdamW | 5 | top-1 accuracy, higher is better | `0.6849` |
| CIFAR-10 ResNet-18, 3 epochs | SGD | 5 | top-1 accuracy, higher is better | `0.2548` |
| CIFAR-10 ResNet-18, 1 epoch, batch 128 | Lion | 3 | top-1 accuracy, higher is better | `0.5794` |
| CIFAR-10 ResNet-18, 1 epoch, batch 128 | NEAT | 3 | top-1 accuracy, higher is better | `0.5538` |
| CIFAR-10 ResNet-18, 1 epoch, batch 128 | AdamW | 3 | top-1 accuracy, higher is better | `0.5507` |
| CIFAR-10 ResNet-18, 1 epoch, batch 128 | SGD | 3 | top-1 accuracy, higher is better | `0.2327` |
| CIFAR-10 ResNet-18, 1 epoch, batch 256 | Lion | 3 | top-1 accuracy, higher is better | `0.5391` |
| CIFAR-10 ResNet-18, 1 epoch, batch 256 | NEAT | 3 | top-1 accuracy, higher is better | `0.5207` |
| CIFAR-10 ResNet-18, 1 epoch, batch 256 | AdamW | 3 | top-1 accuracy, higher is better | `0.5160` |
| CIFAR-10 ResNet-18, 1 epoch, batch 256 | SGD | 3 | top-1 accuracy, higher is better | `0.2361` |
| TinyLlama SST-2 LoRA, 200 train / 100 eval | NEAT | 1 | accuracy, higher is better | `0.52` |
| TinyLlama SST-2 LoRA, 200 train / 100 eval | AdamW | 1 | accuracy, higher is better | `0.52` |
| TinyLlama SST-2 LoRA, 200 train / 100 eval | Lion | 1 | accuracy, higher is better | `0.51` |
| TinyLlama SST-2 LoRA, 200 train / 100 eval | SGD | 1 | accuracy, higher is better | `0.51` |

With the two missing HalfCheetah-v5 cells filled in, the quick RL matrix is now
complete across five seeds. Mean episode reward is high variance and higher is
better:

| Optimizer | Seeds | Mean reward |
|---|---:|---:|
| Lion | 5 | `2311.61` |
| AdamW | 5 | `1873.37` |
| NEAT | 5 | `1333.62` |
| SGD | 5 | `1100.49` |

## Claim boundary

Until the full server suite completes with clean manifests, the current claim is
still: NEAT has reproducible implementations, diagnostics, smoke coverage, and
several completed limited studies. The new CPU results support stability and
rough competitiveness with AdamW in some vision and language quick checks, but
they do not support broad superiority claims. Lion is strongest on several of
the latest quick benchmarks, and NEAT remains slower on CPU.

## Final-suite preparation - 2026-07-31

A validation-locked CIFAR protocol and executable runner were added for the
next clean benchmark commit. This separates a deterministic 10% validation
slice from the CIFAR test set, gives each comparator the same four-trial learning-rate budget,
persists every tuning result, freezes selections in `tuning_manifest.json`, and
then runs the held-out multi-seed test phase. Diffusion evaluation randomness
was also made fixed per seed and epoch so it cannot perturb training RNG state.

The account query on July 31 reported USD `19.11` pending charges and two
unrelated active instances. A Cloud GPU create request remained support-gated.
Attempts to create a new 32-core or 8-core CPU worker were rejected because the
account had reached its monthly fee limit. No existing instance was used,
modified, or deleted. The exact queued studies and capacity-safe dispatch order
are recorded in `vultr-final-validation-suite-2026-07-31.md`.
