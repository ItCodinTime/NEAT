# Benchmark deadline handoff - 2026-08-03

This handoff records the state of the Vultr validation window at
`2026-08-03 11:01 CDT`, with extension updates at `11:15 CDT`, `14:50 CDT`,
`15:28 CDT`, `16:21 CDT`, `16:41 CDT`, `18:46 CDT`, `21:06 CDT`, and
`2026-08-04 13:36 CDT` and `14:21 CDT`. It separates complete evidence,
partial locked-study evidence, CPU-only mimic evidence, and work that remains
queued. Raw Vultr manifests and live JSONL metrics are synchronized under the
ignored `benchmarks/vultr_sync/` directory.

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
| AdamW | 7, 11, 19, 23, 29 | 0.8191, 0.8197, 0.8172, 0.8222, 0.8204 |
| SGD | 7, 11, 19, 23, 29 | 0.8528, 0.8557, 0.8540, 0.8557, 0.8567 |
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

At the `16:41 CDT` update, SGD seeds 23 and 29 completed with test Top-1
`0.8557` and `0.8567`. SGD is now complete for all five held-out seeds with
mean test Top-1 `0.85498`. NEAT remains complete at mean `0.82114`; AdamW and
Lion remain partial. This reinforces that, under the frozen CIFAR-10 ResNet-18
protocol, SGD is the clear winner and NEAT should not be claimed as the vision
leader.

At the `17:19 CDT` update, AdamW seeds 23 and 29 completed with test Top-1
`0.8222` and `0.8204`. AdamW is now complete for all five held-out seeds with
mean test Top-1 `0.81972`. The completed CIFAR-10 comparison for the three
available optimizers is therefore SGD `0.85498`, NEAT `0.82114`, and AdamW
`0.81972`. Lion remains the only missing optimizer in the held-out matrix.

At the `21:06 CDT` deadline status, Lion seeds 7 and 11 were still running on
the four-core CPU worker at epoch `88` with best live metrics `0.8283` and
`0.8269`; no Lion held-out manifest had completed yet. Lion seeds 19, 23, and
29 remain queued behind the TinyLlama proxy on the high-performance worker.

At the `2026-08-04 13:36 CDT` update, Lion seeds 7, 11, 19, and 23 had
completed with test Top-1 `0.8270`, `0.8285`, `0.8218`, and `0.8260`.
Lion seed 29 was still active on both the four-core and high-performance
workers as a deadline recovery duplicate; the high-performance copy was at
epoch `89` with live test Top-1 `0.8305`. The completed Lion mean is therefore
`0.82583` over four seeds only. The completed CIFAR-10 ranking remains SGD
first; Lion is likely second, NEAT narrowly exceeds AdamW, and NEAT is not the
vision leader under this protocol.

At the `14:21 CDT` update, Lion seed 29 completed with test Top-1 `0.8313`.
The full five-seed CIFAR-10 held-out matrix is now complete:

| Optimizer | Mean test Top-1 | Seeds |
|---|---:|---|
| SGD | 0.85498 | 7, 11, 19, 23, 29 |
| Lion | 0.82692 | 7, 11, 19, 23, 29 |
| NEAT | 0.82114 | 7, 11, 19, 23, 29 |
| AdamW | 0.81972 | 7, 11, 19, 23, 29 |

This is a clean negative result for any claim that NEAT is the best CIFAR-10
optimizer under the locked ResNet-18 protocol. NEAT narrowly beats AdamW, but
SGD is the clear winner and Lion also beats NEAT.

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

At the `2026-08-04 13:36 CDT` update, CIFAR-100 NEAT held-out seeds 7, 11, and
19 were active on the high-performance worker. Live epoch-10 test Top-1 values
were `0.4595`, `0.4433`, and `0.4402`; these are progress metrics only, not
completed manifests. Seeds 23 and 29 remain queued behind the first three NEAT
held-out seeds.

At `14:21 CDT`, those NEAT runs were still active at epoch 17 with live Top-1
`0.4691`, `0.4616`, and `0.4648`. A local Mac mini MPS AdamW seed-7 run was
also launched with the frozen AdamW CIFAR-100 LR. It completed 100 epochs with
final test Top-1 `0.5134` and best-epoch Top-1 `0.5135`. The local run is
useful comparator evidence but is marked separately because it uses Apple MPS
rather than the Vultr CPU environment.

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

An additional local Mac mini CPU sweep completed on `2026-08-04`: conflict
intensities `0.25`, `0.5`, `0.75`, and `0.9`; noise levels `0.0`, `0.1`, and
`0.2`; optimizers NEAT, SGD, PCGrad, and CAGrad; seeds 7, 11, and 19; 1,000
steps and dimension 64. This produced `144` result manifests. SGD had the best
mean final distance in all 12 conflict/noise cells. NEAT was consistently close
to SGD and generally far better than PCGrad and CAGrad in noisy high-conflict
settings, but the broader sweep reinforces the same claim boundary: NEAT has a
controlled conflict-handling advantage over PCGrad/CAGrad in this testbed, not
a win over plain SGD.

The earlier full HalfCheetah-v5 queue did not produce a completed manifest
before worker consolidation. A smaller CPU proxy completed on the extra
two-core worker: HalfCheetah-v5, 20,000 environment steps, 5,000-step
evaluation cadence, five evaluation episodes, seed 7, NEAT versus AdamW.
Higher mean episode reward is better.

| Optimizer | Final reward | Best checkpoint reward | Status |
|---|---:|---:|---|
| NEAT | -217.2492 | -27.6415 | preliminary CPU proxy |
| AdamW | -63.5017 | -63.5017 | preliminary CPU proxy |

This is single-seed deadline evidence, not a multi-seed MuJoCo matrix.

Additional 20,000-step HalfCheetah-v5 CPU proxy seeds completed after the
original checkpoint:

| Seed | NEAT final/best reward | AdamW final/best reward | Read |
|---:|---:|---:|---|
| 7 | -217.2492 / -27.6415 | -63.5017 / -63.5017 | Mixed; NEAT best checkpoint higher, AdamW final higher |
| 11 | 46.6699 / 46.6699 | 40.1066 / 40.1066 | NEAT slightly higher |
| 19 | 62.0183 / 62.0183 | pending in synchronized docs at this checkpoint | Incomplete pair |
| 23 | -17.8536 / -17.8536 | 114.6506 / 114.6506 | AdamW higher |

Seeds 29 and 31 for both NEAT and AdamW were launched on the extra two-core
worker at `2026-08-04 13:33 CDT`. They are deadline CPU proxies and remain
preliminary until the manifests complete and are synchronized.

At `14:21 CDT`, the seed 29 and 31 HalfCheetah-v5 proxy manifests had
completed and were synchronized. Across the expanded synchronized seed set,
using final reward, NEAT wins seeds 11 and 19 while AdamW wins seeds 7, 23,
29, and 31. Using best checkpoint reward, the read is closer to split: NEAT
wins seeds 7, 11, and 19; AdamW wins seeds 23, 29, and 31. The larger CPU RL
proxy is therefore mixed and does not establish an RL advantage for NEAT.

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

The deadline queue also includes CPU proxies requested after the GPU
entitlement check failed:

| Proxy | Scope | Status at 16:21 CDT |
|---|---|---|
| SST-2 | DistilBERT, 2,048 training examples, 500 eval examples, seed 7, NEAT vs AdamW | Complete; NEAT `0.872`, AdamW `0.872` |
| Alpaca | TinyLlama 1.1B Chat, LoRA, 64 training examples, 16 eval examples, seed 7, NEAT vs AdamW | Running on the high-performance worker after fixing a `torchvision`/Llama import conflict |
| HalfCheetah | 20,000-step CPU MuJoCo proxy, seed 7, NEAT vs AdamW | Complete; NEAT best `-27.6415`, AdamW best `-63.5017` |

These proxies are not substitutes for full GLUE, full instruction tuning,
Atari, or a multi-seed MuJoCo matrix. They exist to produce deadline-limited
stress evidence on CPU-only infrastructure.

At `16:21 CDT`, the high-performance worker queue was changed to skip completed
SST-2 and HalfCheetah duplicates. It now runs TinyLlama LoRA first after the
active CIFAR-10 AdamW/SGD recovery, then Lion seeds 19, 23, and 29, then the
CIFAR-100 NEAT held-out seeds if time remains.

The first TinyLlama launch failed because `transformers` imported
`torchvision`, which raised a CPU-wheel `torchvision::nms` registration error
while loading `LlamaConfig`. Since CIFAR-10 AdamW/SGD had completed and no HPC
vision process was active, `torchvision` was temporarily removed for the Llama
proxy and is reinstalled by the queue before Lion and CIFAR-100 vision resume.
The corrected TinyLlama NEAT run was active at `18:46 CDT`.

At the `21:06 CDT` deadline status, both TinyLlama LoRA processes were still
active with no completed manifest: NEAT had been running for about `2h20m`,
and a parallel AdamW comparator for about `1h43m`. They were CPU-bound with
stable memory use, so they were left running rather than interrupted.

At the `2026-08-04 13:36 CDT` update, the TinyLlama LoRA proxy had completed:
NEAT validation perplexity `7.8410`, AdamW validation perplexity `7.8253`
where lower is better. AdamW is slightly better on this small CPU
instruction-tuning proxy. The result is useful as a deadline smoke test, but
the sample is too small to support a Llama/Mistral instruction-tuning claim.

The local Mac mini also completed an STS-style CPU mimic matrix:
Multi-MNIST and Multi-Fashion-MNIST, two seeds, five epochs, 5,000 train
examples, 1,000 validation examples, Adam, NEAT, PCGrad, and MGDA. Mean
validation average accuracy was:

| Dataset | Adam | NEAT | PCGrad | MGDA |
|---|---:|---:|---:|---:|
| Multi-MNIST | 0.7958 | 0.0908 | 0.8000 | 0.7830 |
| Multi-Fashion-MNIST | 0.6365 | 0.0978 | 0.6363 | 0.6307 |

This is intentionally labeled as an untuned CPU mimic. It does not show NEAT
strength; it shows that the current STS runner plus shared quick learning rate
is a poor expression of NEAT and needs a validation-tuned NEAT setting before
being used for any positive multi-task claim.

A follow-up local NEAT-only STS learning-rate sweep tested `0.01`, `0.003`,
`0.001`, `0.0003`, and `0.0001` for both Multi-MNIST and Multi-Fashion-MNIST
with 10 epochs, 10,000 training examples, and 2,000 validation examples. The
best final validation average accuracy remained near chance: `0.1040` on
Multi-MNIST and `0.1090` on Multi-Fashion-MNIST. This suggests the STS failure
is not fixed by this simple LR sweep.

A larger local SST-2 DistilBERT 10,000-example CPU run was attempted, but it
was stopped before the first epoch metric because it slowed the higher-value
local CIFAR-100 MPS run substantially. No SST-2 result is recorded from that
attempt.

## Capacity record

After consolidation and follow-on provisioning, three NEAT-labelled workers are
actively computing: one four-core general compute worker, one eight-core
high-performance worker, and one extra two-core general compute worker. One
additional two-core high-performance worker was accepted by the Vultr create
API, but entered an `active`/`stopped`/`locked` state and failed an explicit
start request with a provider-side HTTP 500 response:

| Instance | Plan | Observed state |
|---|---|---|
| `cfbf10fe-2cf6-46ef-bf97-259913043dee` | `vhp-2c-4gb-intel` | Accepted at `14:45 CDT`; locked with no usable SSH path |
| `c2f497a0-8754-487d-b9e3-d15e9132a454` | `vc2-2c-4gb` | Accepted at `15:15 CDT`; recovered to `running ok` and assigned to the HalfCheetah proxy |

Larger 8-core and 12-core follow-on create attempts were blocked by the
monthly-fee quota or Vultr backend plan errors. Further extra worker creation
is therefore paused unless the provider unlocks or releases the remaining stuck
instance. Two unrelated instances were first snapshotted and then destroyed at
the user's direction; their snapshots and recovery commands are stored in a
private local recovery note outside this repository. The provider's
monthly-fee quota and GPU-approval requirement prevented any accepted GPU
deployment.

The active Vultr serverless inference endpoint is not part of the optimizer
benchmark execution. These benchmark runs use CPU SSH workers and local CPU
execution; serverless inference can be terminated if it is unrelated to NEAT.

## Remaining work

1. Finish the active CIFAR-100 NEAT held-out seeds, then run held-out
   CIFAR-100 seeds for AdamW, SGD, and Lion if CPU time remains.
2. Add and execute the acceptance-gated GPU protocols in the
   [industry benchmark matrix](industry-benchmark-matrix.md): ImageNet-1K,
   full GLUE, language-model pretraining, Atari, real multi-task datasets, and
   Llama/Mistral instruction tuning.

No result beyond the explicitly listed completed cells should be described as
finished or industry-scale evidence. As of the `2026-08-04 13:36 CDT` update,
NEAT has credible narrow evidence for controlled synthetic conflict handling
versus PCGrad/CAGrad, is roughly competitive with AdamW on CIFAR-10 but far
behind SGD, slightly loses the TinyLlama and SST-2 CPU proxies to AdamW or ties,
and has mixed preliminary HalfCheetah CPU evidence. It has not demonstrated an
industry-scale win on ImageNet-1K, full GLUE, language-model pretraining,
Atari, NYUv2/MT50, or Llama/Mistral instruction tuning.
