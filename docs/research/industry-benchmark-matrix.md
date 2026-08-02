# Industry benchmark matrix

This matrix maps the benchmark program to commonly expected optimizer evidence
categories. It distinguishes implemented runners, reproducible pending tracks,
and work that must be implemented before it can be claimed. A row is only
"complete" when it has a locked protocol where appropriate, a clean source
commit, raw manifests, multi-seed aggregates, and a documented comparator
table.

## Evidence status

| Category | Benchmark | NEAT runner status | Minimum credible protocol | Current status |
|---|---|---|---|---|
| Image classification | CIFAR-10 | `torch_suite.locked_study` | ResNet-18; equal four-LR validation search; five locked test seeds | Running; see the [live ledger](vultr-locked-run-ledger-2026-08-02.md) |
| Image classification | CIFAR-100 | `torch_suite.locked_study` | ResNet-34; equal four-LR validation search; five locked test seeds | Implemented, pending capacity |
| Large vision | ImageNet-1K | `torch_suite.vision` via ImageFolder | ResNet-50 or ViT; GPU; 90 epochs; five seeds; dataset license and checksum record | Runner supports ImageFolder; full protocol and GPU capacity are pending |
| NLP fine-tuning | GLUE | `torch_suite.language_model` supports SST-2 | Fixed backbone and tokenization; matched search budget; report each GLUE task and macro aggregate | SST-2 implemented; full GLUE task matrix pending implementation |
| Language modeling | WikiText-103 / C4 subset | No pretraining runner | Fixed tokenizer, model, token budget, sequence length, and held-out perplexity | Pending implementation and GPU capacity |
| Reinforcement learning | MuJoCo | `torch_suite.reinforcement_learning` | At least three environments, five seeds, fixed step budget, return mean and variance | Runner implemented; only quick evidence exists |
| Reinforcement learning | Atari | No Atari runner | Standard preprocessing, fixed frames, five seeds, score normalization and human/random references | Pending implementation and GPU capacity |
| Multi-task learning | Synthetic conflict | `torch_suite.multitask_synthetic` | PCGrad/CAGrad/SGD/NEAT, known optimum, five seeds | Implemented; synthetic-only evidence |
| Multi-task learning | MultiMNIST / NYUv2 / MT50 | No task-gradient runner | Shared trunk; per-task losses and gradients; PCGrad/CAGrad/SGD/NEAT; task and aggregate metrics | Pending implementation; MT50 also needs substantial simulator capacity |
| Instruction fine-tuning | Alpaca-style | `torch_suite.language_model --task alpaca --lora` | Licensed dataset/model, frozen prompt template, LoRA target modules, held-out generation-quality evaluator | Preliminary runner exists; credible Llama/Mistral study is pending GPU, licensing, and evaluator work |

## Acceptance requirements

Every future category must record the following before results are promoted:

1. Dataset source, version, license status, checksum, split definition, and
   preprocessing.
2. Exact model checkpoint and tokenizer revision, trainable parameter count,
   precision policy, effective batch size, and learning-rate schedule.
3. A pre-registered optimizer search budget that is equal in trial count and
   objective. Test metrics must not choose settings.
4. At least five independent seeds for headline comparisons, with paired
   uncertainty intervals and all raw manifests available.
5. Hardware model, accelerator count, software versions, wall time, peak
   memory, and a cost record.
6. NEAT diagnostics plus the standard task metric. For multi-task studies,
   retain the per-task gradient representation used by every conflict-aware
   comparator.

## Dispatch order after the locked CIFAR study

1. Complete the CIFAR-10 held-out matrix and publish it only if all expected
   seeds exist.
2. Run the CIFAR-100 locked study on a GPU-capable worker; this is the next
   high-signal vision transfer test.
3. Run full MuJoCo across HalfCheetah, Hopper, and Walker2d with five seeds;
   report seed variance and area under the learning curve.
4. Extend the language runner from SST-2 to a fixed documented GLUE subset,
   then run a small C4 or WikiText pretraining protocol with held-out
   perplexity.
5. Add a real multi-task task-gradient harness, beginning with MultiMNIST,
   before scheduling NYUv2 or MT50.
6. Add the licensed Llama or Mistral LoRA instruction-tuning protocol and its
   held-out evaluator.
7. Schedule ImageNet-1K and Atari only after GPU provisioning is available;
   neither is a credible CPU-deadline workload.

## Current 24-hour capacity decision

The active Vultr account has prepaid credit but rejects additional deployment
with a monthly-fee-limit error; Cloud GPU provisioning also requires provider
approval. On 2026-08-02, the unrelated `Cluster Node` and
`qrouter-worker-20260731` instances were paused at the user's direction, but
their provisioned plans still count toward this quota. The two accepted
four-core CPU workers are therefore reserved for the already-running
validation-locked CIFAR-10 comparison. Replacing that workload with tiny
one-seed stand-ins for ImageNet, full GLUE, pretraining, Atari, MT50, or
Llama/Mistral would not satisfy this matrix and would reduce the quality of the
only headline-grade run in progress.

The program is now fully specified and documented, but only artifacts that
meet the acceptance requirements may be described as benchmark results.
