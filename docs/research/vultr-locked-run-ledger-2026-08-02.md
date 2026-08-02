# Vultr locked-run ledger - 2026-08-02

This is the live execution ledger for the final CPU validation window. It is
the authoritative status record for the two validation-locked CIFAR studies;
it complements, rather than replaces, the pre-registered suite and protocol.

## Claim boundary

Only a complete, clean-commit matrix of five held-out test seeds for each of
NEAT, AdamW, SGD, and Lion may be included in a strict paper-style aggregate.
An incomplete matrix is useful operational evidence, but must be labelled
**partial** and must not be used to claim an optimizer ranking. Tuning trials
are never test evidence.

## Reproducibility record

| Field | Value |
|---|---|
| Source commit | `55c3df515e30c32fd70ae2c139c8243593d20464` |
| Dataset | CIFAR-10 archive copied from the checksum-validated local mirror |
| Study | CIFAR-10 / ResNet-18 |
| Validation split | 10% of training data, `split_seed=20260731` |
| Tuning seed | `101` |
| Held-out test seeds | `7`, `11`, `19`, `23`, `29` |
| Training recipe | 100 epochs, batch size 128, weight decay 0.0005 |
| Comparator set | NEAT, AdamW, SGD, Lion |
| Test selection rule | Settings frozen in `tuning_manifest.json` before test runs |
| Worker class | Vultr `vc2-4c-8gb`, CPU-only PyTorch |
| Artifact handling | Remote logs and manifests synchronize before timed cleanup |

The study code stores the selected-setting manifest beside the raw tuning
trials and annotates every held-out result with the split, tuning state, study
identifier, and manifest path. The implementation is described in the
[validation-locked protocol](validation-locked-benchmark-protocol.md).

## Frozen tuning selections

Every optimizer received four learning-rate candidates and the same 15-epoch
validation budget. The selections below were made before test evaluation
started.

| Optimizer | Selected learning rate | Final validation Top-1 |
|---|---:|---:|
| NEAT | 0.003 | 0.8018 |
| AdamW | 0.001 | 0.7938 |
| SGD | 0.1 | 0.8194 |
| Lion | 0.0003 | 0.7980 |

The full candidate table is retained in the remote and synchronized
`tuning_manifest.json`; no held-out test accuracy was used to choose these
values.

## Dispatch and current status

The primary worker executes the registered study serially. A second worker
uses the same commit, data archive, split, selected-setting manifest, and
command-line recipe to run independent AdamW and SGD seeds two at a time.
This sharding changes throughput only; it does not change a trial's model,
seed, metric, or frozen setting.

| Work item | Status at 2026-08-02 12:07 CDT |
|---|---|
| CIFAR-10 validation tuning | Complete: 16 of 16 trials |
| CIFAR-10 NEAT held-out tests | Complete for seeds 7, 11, and 19; seed 23 in progress |
| CIFAR-10 AdamW held-out tests | Seeds 7 and 11 launched on the shard worker |
| CIFAR-10 SGD held-out tests | Seeds 7 and 11 launched on the shard worker |
| CIFAR-10 Lion held-out tests | Queued after the AdamW and SGD comparison lanes |
| CIFAR-100 / ResNet-34 | Not started; no result should be implied |

The three completed held-out NEAT trials report Top-1 values of `0.8249`,
`0.8211`, and `0.8213` for seeds 7, 11, and 19 respectively. These are
partial held-out observations, not a completed comparator result.

## Capacity and cost record

At dispatch, the Vultr account reported a negative balance (prepaid credit)
and USD `9.44` pending charges. A request for an additional six-core,
16 GB worker was rejected by the provider with: "You have reached the maximum
monthly fee limit for this account." This is a provider-side deployment quota,
separate from the available credit. GPU creation remains subject to a support
approval requirement.

The run therefore uses the two newly-created benchmark workers already
accepted by the provider. At `2026-08-02 12:20 CDT`, the user directed that
the unrelated `Cluster Node` and `qrouter-worker-20260731` instances be put on
hold. Their `power_status` is now `stopped`; neither was repurposed or deleted.
A fresh six-core NEAT worker request was retried after the pause and received
the same monthly-fee-limit rejection. Vultr therefore counts these provisioned
but stopped plans toward the deployment quota. The watchdog limit is USD `120`
pending charges and it copies artifacts before deleting a completed or
timed-out worker.

## Reporting rules for the deadline handoff

1. Synchronize all manifests, logs, and selection files from both workers.
2. Re-run the repository lint, full tests, and strict documentation build on
   the final source tree.
3. Use the strict paper report only if every expected held-out manifest exists.
4. Otherwise publish a partial-results table with seed counts, wall time, and
   the reason the matrix is incomplete; do not calculate a headline ranking.
5. Record the provider quota response and outstanding queue so a later run can
   resume without silently changing the protocol.
