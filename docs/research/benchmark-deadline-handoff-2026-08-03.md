# Benchmark deadline handoff - 2026-08-03

This handoff records the state of the Vultr validation window at
`2026-08-03 11:01 CDT`, with extension updates at `11:15 CDT` and
`14:50 CDT`. It separates complete evidence, partial locked-study evidence,
and work that remains queued. Raw manifests and live JSONL metrics are
synchronized under the ignored `benchmarks/vultr_sync/` directory.

## Quality gates

The source revision for the original validation protocol is
`55c3df515e30c32fd70ae2c139c8243593d20464`. The current repository revision
for deadline execution is `812e78d`, which includes the idempotent locked-study
resume fix. Before the run, local checks passed: Ruff, `111` tests with three
expected TensorFlow skips, and the strict MkDocs build. The validation-locked
mechanism, its manifest tests, and the protocol are documented in the
[locked-study protocol](validation-locked-benchmark-protocol.md).

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

At the `14:50 CDT` update, the remaining AdamW and SGD CIFAR-10 jobs are
running on the eight-core high-performance worker. The live JSONL metrics show
AdamW seeds 23 and 29 around epoch `55`, and SGD seeds 23 and 29 around epochs
`63` to `64`. Lion seeds 7 and 11 are running on the four-core CPU worker and
are around epoch `32`; Lion seeds 19, 23, and 29 are queued on the
high-performance worker after the AdamW/SGD recovery finishes. These are live
progress metrics only, not completed held-out manifests.

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

The deterministic 64-dimensional synthetic conflict suite is now complete
for NEAT, PCGrad, CAGrad, and SGD: five seeds per optimizer, 1,000 steps,
dimension 64, conflict intensity `0.75`, and noise `0.1`. The outcome is
distance to the known optimum; lower is better.

| Optimizer | Seeds | Mean final distance | Range |
|---|---:|---:|---:|
| SGD | 5 | 0.03872 | 0.03251 to 0.04553 |
| NEAT | 5 | 0.04307 | 0.03625 to 0.05019 |
| CAGrad | 5 | 0.31807 | 0.31498 to 0.32018 |
| PCGrad | 5 | 1.10532 | 1.09653 to 1.11619 |

The initial CAGrad dependency failure is retained in the worker log. After
installing its declared SciPy extra, the missing CAGrad and SGD cells were
rerun with the original fixed configuration. NEAT is substantially better
than PCGrad and CAGrad in this controlled task, but SGD is better than NEAT;
this result must not be represented as general multi-task superiority.

The earlier full HalfCheetah-v5 queue did not produce a completed manifest
before worker consolidation. A smaller CPU proxy is queued on the
high-performance worker after the CIFAR-10 Lion recovery: HalfCheetah-v5,
20,000 environment steps, 5,000-step evaluation cadence, five evaluation
episodes, seed 7, NEAT versus AdamW. It is a preliminary CPU proxy and remains
separate from vision claims.

## Extension execution notes

The original CIFAR-10 `all` process was confirmed stalled: it had no active
child computation and no new artifacts for nearly three days. It was stopped
and its worker reassigned to the missing Lion held-out cells. The replacement
uses the same frozen Lion learning rate, test split, selection manifest,
seeds, and thread budget as the locked study.

This recovery also identified that evaluation resumption replayed completed
trial directories. Commit `fa3f33a` makes resumption idempotent by skipping a
trial only when its directory contains exactly one completed artifact; the
behavior is unit tested. Existing artifacts retain the original protocol
revision recorded above.

The deadline queue also includes two additional CPU proxies requested after the
GPU entitlement check failed:

| Proxy | Scope | Status at 14:50 CDT |
|---|---|---|
| SST-2 | DistilBERT, 2,048 training examples, 500 eval examples, seed 7, NEAT vs AdamW | Queued after CIFAR-10 Lion on the high-performance worker |
| Alpaca | TinyLlama 1.1B Chat, LoRA, 64 training examples, 16 eval examples, seed 7, NEAT vs AdamW | Queued after SST-2 on the high-performance worker |
| HalfCheetah | 20,000-step CPU MuJoCo proxy, seed 7, NEAT vs AdamW | Queued after Alpaca on the high-performance worker |

These proxies are not substitutes for full GLUE, full instruction tuning,
Atari, or a multi-seed MuJoCo matrix. They exist to produce deadline-limited
stress evidence on CPU-only infrastructure.

## Capacity record

After consolidation, two NEAT-labelled workers are actively computing: one
four-core general compute worker and one eight-core high-performance worker.
Two extra small workers were accepted by the Vultr create API, but both entered
`active`/`stopped`/`locked` states and failed explicit start requests with
provider-side HTTP 500 responses:

| Instance | Plan | Observed state |
|---|---|---|
| `cfbf10fe-2cf6-46ef-bf97-259913043dee` | `vhp-2c-4gb-intel` | Accepted at `14:45 CDT`; locked with no usable SSH path |
| `c2f497a0-8754-487d-b9e3-d15e9132a454` | `vc2-2c-4gb` | Accepted at `15:15 CDT`; locked/stopped immediately after IP assignment |

Larger 8-core and 12-core follow-on create attempts were blocked by the
monthly-fee quota or Vultr backend plan errors. Extra worker creation is
therefore paused until the provider unlocks or releases those instances. Two
unrelated instances were first snapshotted and then destroyed at the user's
direction; their snapshots and recovery commands are stored in a private local
recovery note outside this repository. The provider's monthly-fee quota and
GPU-approval requirement prevented any accepted GPU deployment.

## Remaining work

1. Finish the CIFAR-10 held-out matrix: AdamW seeds 23 and 29, SGD seeds 23
   and 29, and all five Lion seeds.
2. Run five held-out CIFAR-100 seeds for every frozen optimizer setting.
3. Execute and label the queued CPU proxies for SST-2, TinyLlama LoRA, and
   HalfCheetah-v5.
4. Add and execute the acceptance-gated GPU protocols in the
   [industry benchmark matrix](industry-benchmark-matrix.md): ImageNet-1K,
   full GLUE, language-model pretraining, Atari, real multi-task datasets, and
   Llama/Mistral instruction tuning.

No result beyond the explicitly listed completed cells should be described as
finished or industry-scale evidence.
