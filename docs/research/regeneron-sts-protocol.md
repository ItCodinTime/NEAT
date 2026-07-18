# Regeneron STS multi-task protocol

## Research question

The preregistered hypothesis is that NEAT's conflict-aware correction should
improve shared-representation learning when optimization exhibits conflict. The
study tests accuracy, mechanism signals, and overhead. A positive average alone
is not treated as evidence: all headline comparisons use matched seeds, paired
tests, confidence intervals, and effect sizes.

The mechanism plots are association and dose-response evidence. They do not, by
themselves, establish causation. The correction-off ablation is the direct
intervention used to strengthen the mechanism argument.

## Datasets

`Multi-MNIST` and `Multi-Fashion-MNIST` are generated from the official
torchvision training and test archives. Two source images are placed on a
36-by-36 single-channel canvas. Task 1 begins at `(0, 0)`, task 2 begins at
`(8, 8)`, and overlapping pixels use their maximum intensity. The heads predict
the labels of the first and second source positions, respectively.

The official 60,000-example training archive is split into 50,000 training and
10,000 validation examples with seed `20260718`. The official 10,000-example
test archive remains the final test set. Pairings use fixed seeds `314159`,
`271828`, and `161803` for train, validation, and test. Optimizer seeds never
change examples, pairings, or splits. The machine-readable dataset specification
hash is saved in every run.

This fixed-position construction is fully reproducible but is one specific
Multi-MNIST variant. Claims must name it and must not imply that results cover
every published Multi-MNIST generation procedure.

## Model and training controls

Every method receives the same randomly initialized two-head CNN for each seed.
The shared trunk has three convolutional layers, two max-pooling stages,
adaptive 4-by-4 pooling, and a 128-unit shared layer. Each task has a separate
10-class linear head. The runner records the exact parameter count.

The two cross-entropy losses are averaged, so changing from one to two tasks
does not double the nominal loss scale. Batch order, batch size, epoch count,
cosine learning-rate schedule, data split, precision, deterministic policy, and
hardware are held fixed across methods. Weight decay defaults to zero. Final
runs use seeds `7, 11, 19, 23, 29`.

## Compared methods

- **Adam** uses `torch.optim.Adam` on the mean task loss.
- **PCGrad** computes both task gradients, projects each only when their global
  dot product is negative, averages the projected gradients, and uses Adam as
  the transport optimizer.
- **MGDA** computes the exact minimum-norm convex combination for two task
  gradients and uses Adam as the transport optimizer. No gradient normalization
  is applied.
- **NEAT base** uses the Nash correction with `alpha=0.25`, `beta=0.9`, and the
  momentum opponent. Adaptive alpha, adaptive preconditioning, Lion transport,
  Lookahead, gradient centralization, Nesterov momentum, conflict thresholding,
  and warmup are all off.

Learning rates are tuned separately for each dataset-method pair over the same
declared grid. Selection maximizes the mean final-three-epoch validation average
accuracy over tuning seeds `101` and `103`. Tuning runs pass `--skip-test`; the
test split is not loaded into the selection calculation. The selected mapping is
locked in `locked_lrs.json` before Tier 1 starts.

## Metrics and statistics

Each epoch records training loss, loss variance, both validation accuracies,
average validation accuracy, task-gradient cosine on a held-out diagnostic
batch, correction ratio, gradient-update alignment, learning rate, training
time, evaluation time, and diagnostic time.

The final table reports task 1, task 2, and average test accuracy as mean,
sample standard deviation, and two-sided 95% t confidence interval over five
seeds. NEAT is compared with every baseline using a paired t-test, Wilcoxon
signed-rank test, mean paired difference, and paired Cohen's d. Exact values are
saved as JSON and CSV; rounded paper tables are generated from those files.

Correction metrics have method-specific meanings and are labeled in each
manifest. Adam is the identity correction. PCGrad and MGDA compare their
task-surgery gradient with the raw task mean. NEAT compares its Nash-corrected
gradient with the raw gradient before momentum transport. Cross-method absolute
correction ratios therefore require care; the primary mechanism comparison is
NEAT across matched single-task and multi-task settings.

## Mechanism and ablations

For each dataset and paper seed, NEAT runs both single-task heads independently
using the same paired inputs and model family. The first mechanism plot compares
single-task and multi-task correction ratio by epoch. The second compares each
NEAT run's mean correction ratio with its paired accuracy gain over Adam.

One-factor ablations start from NEAT base and change exactly one named option:
correction off, adaptive alpha, adaptive preconditioning, previous-gradient
opponent, gradient-EMA opponent, conflict threshold `0.05`, one-epoch correction
warmup, Lion transport, Lookahead with `k=5`, gradient centralization, or
Nesterov momentum. The correction-off result is the most important mechanism
control. No multi-flag adaptive stack is called “base NEAT.”

A conflict-severity sweep is optional until Tier 1, mechanism runs, and the
correction-off ablation are complete. Its generation rule and levels must be
committed before results are inspected.

## Cost, provenance, and stopping rules

The approved Vultr ceiling is USD 100, with a stop-and-review point at USD 80.
The current fallback instance is `vhf-8c-32gb` in New Jersey at USD 0.263 per
hour. Cloud GPU provisioning was attempted first and rejected because the
account requires product approval; no support request was submitted. The CPU
fallback is used consistently unless a new hardware phase is explicitly
declared.

Every final run must record a clean git commit, exact command, Python and package
versions, OS, CPU count, device, deterministic policy, seeds, optimizer config,
dataset specification hash, per-epoch history, final metrics, training and wall
time, and peak process memory. `benchmarks.sts.environment` additionally records
CPU details, disk/memory state, `pip freeze`, and SHA-256 hashes of downloaded
dataset files. Secrets are never copied into experiment artifacts.

Stop conditions are NaN/Inf metrics, missing or duplicate seed cells, dataset
hash disagreement, dirty or unknown git state for final runs, test access during
tuning, an unexplained accuracy regression in the learning sanity check, or the
USD 80 review threshold. Completed raw results are never edited; corrected runs
receive a new committed protocol and are reported with the reason for exclusion.

## Negative results and claim policy

The existing digits benchmark where NEAT trails SGD and Adam remains part of the
limitations evidence. New null or negative findings are retained. A result is
not described as significant when `p >= 0.05`, and a non-significant result is
not reframed as “better on average.” Multiple comparisons, the small five-seed
sample, one model family, one dataset-construction variant, and CPU-only hardware
are explicit limitations.
