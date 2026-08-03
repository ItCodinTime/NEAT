# Benchmark deadline handoff - 2026-08-03

This handoff records the state of the Vultr validation window at
`2026-08-03 11:01 CDT`. It separates complete evidence, partial locked-study
evidence, and work that remains queued. Raw manifests are synchronized under
the ignored `benchmarks/vultr_sync/` directory.

## Quality gates

The source revision for the validation protocol is
`55c3df515e30c32fd70ae2c139c8243593d20464`. Before the run, local checks
passed: Ruff, `110` tests with three expected TensorFlow skips, and the strict
MkDocs build. The validation-locked mechanism, its manifest tests, and the
protocol are documented in the [locked-study protocol](validation-locked-benchmark-protocol.md).

## CIFAR-10 ResNet-18 locked study

The equal-budget validation search is complete: all four optimizers received
four learning-rate trials and settings were frozen before test evaluation.

| Optimizer | Frozen LR | Validation Top-1 |
|---|---:|---:|
| NEAT | 0.003 | 0.8018 |
| AdamW | 0.001 | 0.7938 |
| SGD | 0.1 | 0.8194 |
| Lion | 0.0003 | 0.7980 |

Held-out test evidence at the checkpoint is partial:

| Optimizer | Completed unique seeds | Test Top-1 values |
|---|---|---|
| NEAT | 7, 11, 19, 23, 29 | 0.8249, 0.8211, 0.8213, 0.8189, 0.8195 |
| AdamW | 7, 11, 19 | 0.8191, 0.8197, 0.8172 |
| SGD | 7, 11, 19 | 0.8528, 0.8557, 0.8540 |
| Lion | none | pending |

Seed 7 for AdamW was duplicated across two workers during recovery sharding.
The table retains one manifest only and does not treat it as an extra
independent seed. The matrix is therefore not complete and must not receive a
strict paper aggregate or a headline optimizer ranking. It does show that the
NEAT test runs are stable in the `0.819` to `0.825` range under the frozen
recipe; it does not establish comparative superiority.

## CIFAR-100 ResNet-34 locked study

All 16 equal-budget validation trials completed. The selected settings are:

| Optimizer | Frozen LR | Validation Top-1 |
|---|---:|---:|
| NEAT | 0.003 | 0.4890 |
| AdamW | 0.001 | 0.4766 |
| SGD | 0.1 | 0.5054 |
| Lion | 0.0003 | 0.4852 |

No CIFAR-100 held-out test run had completed at the checkpoint, so these are
tuning observations only. They are not test-set evidence.

## Multi-task and control evidence

The deterministic 64-dimensional synthetic conflict suite completed five
seeds for NEAT and PCGrad at 1,000 steps. The CAGrad run stopped because the
worker lacked SciPy; this dependency failure is retained in the log and the
remaining CAGrad and SGD cells are pending rerun. This category remains
separate from vision claims. The full MuJoCo queue did not start before the
deadline; earlier quick MuJoCo results remain documented as preliminary only.

## Capacity record

Four NEAT-labelled Vultr CPU workers were active at the checkpoint: three
four-core workers and one two-core worker. Two unrelated instances were first
snapshotted and then destroyed at the user's direction; their snapshots and
recovery commands are stored in a private local recovery note outside this
repository. The account reported USD `13.86` pending charges at the checkpoint.
The provider's monthly-fee quota and GPU-approval requirement prevented any
additional GPU or higher-capacity deployment.

## Remaining work

1. Finish the CIFAR-10 held-out matrix: AdamW seeds 23 and 29, SGD seeds 23
   and 29, and all five Lion seeds.
2. Run five held-out CIFAR-100 seeds for every frozen optimizer setting.
3. Finish the CAGrad and SGD synthetic cells, then run the full multi-seed
   MuJoCo matrix.
4. Add and execute the acceptance-gated GPU protocols in the
   [industry benchmark matrix](industry-benchmark-matrix.md): ImageNet-1K,
   full GLUE, language-model pretraining, Atari, real multi-task datasets, and
   Llama/Mistral instruction tuning.

No result beyond the explicitly listed completed cells should be described as
finished or industry-scale evidence.
