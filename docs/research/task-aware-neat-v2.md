# Task-aware NEAT v2

## Status and motivation

The first STS study is a completed negative result. Base NEAT received the mean
task gradient, measured conflict against historical momentum, and used its own
momentum transport. It therefore did not observe the task-to-task gradient
conflict used by PCGrad and MGDA. The correction-off result also collapsed,
while sign transport nearly recovered baseline accuracy. Those controls point
to transport and learning-rate mismatch as the main cause of the collapse.

NEAT v2 is an exploratory method revision. Its results must not be pooled with
or presented as confirmation of the preregistered v1 hypothesis.

## Update rule

For task gradients `g_1, ..., g_k` on the shared model parameters, define each
task's opponent as the mean of the other task gradients:

```text
o_i = mean_{j != i}(g_j)
```

When `g_i` and `o_i` have a negative global dot product, v2 applies a bounded
soft projection:

```text
g_i' = g_i - alpha * proj_{o_i}(g_i)   if dot(g_i, o_i) < 0
g_i' = g_i                              otherwise
g     = mean_i(g_i')
```

`alpha` is restricted to `[0, 1]`. The projection itself already scales with
the conflict angle, so v2 does not multiply by negative cosine a second time.
This changes the correction magnitude from approximately
`alpha * cosine^2` to `alpha * abs(cosine)`.

Conflict is measured once over the flattened shared trunk. Task-specific head
gradients are averaged without correction. The combined gradient is then
passed to an unchanged Adam or published Lion transport. NEAT is consequently
a gradient transformation, not a replacement transport optimizer.

## Required controls

- `adam_alpha_0` must reproduce the Adam direction while retaining the same
  per-task-gradient measurement overhead as corrected v2.
- `lion_alpha_0` is the corresponding Lion control.
- `alpha` values `0`, `0.25`, `0.5`, and `1` separate a dose response from a
  transport effect.
- Plain Adam and plain Lion measure the absolute compute overhead of obtaining
  separate task gradients.
- A later norm-matched random-direction control is required before correction
  geometry is described as causal.

The initial promotion gate uses tuning seeds `101` and `103`, five epochs, and
validation metrics only:

```bash
python -m benchmarks.sts.run_suite promotion \
  --seeds 101 103 \
  --epochs 5 \
  --output-dir benchmark-runs/sts-v2-gate
```

A corrected candidate advances only if its paired validation gain over
`adam_alpha_0` is positive on both datasets, neither seed collapses, and the
measured correction is non-zero. Advancing a candidate triggers a separate
method-specific learning-rate tune before any five-seed test run. Failed gate
results remain documented as negative evidence.

## Promotion result: 2026-07-20

The validation-only gate completed all 20 declared trials on Vultr at clean git
revision `32ef65f`. It used two tuning seeds, five epochs, and did not evaluate
the test split.

| Dataset | Alpha | Control | Candidate | Paired gain | Correction ratio |
|---|---:|---:|---:|---:|---:|
| Multi-MNIST | 0.25 | 0.9630 | 0.9630 | -0.0000 | 0.0207 |
| Multi-MNIST | 0.50 | 0.9630 | 0.9641 | +0.0011 | 0.0378 |
| Multi-MNIST | 1.00 | 0.9630 | 0.9630 | +0.0000 | 0.0749 |
| Multi-Fashion-MNIST | 0.25 | 0.8467 | 0.8442 | -0.0025 | 0.0183 |
| Multi-Fashion-MNIST | 0.50 | 0.8467 | 0.8463 | -0.0005 | 0.0367 |
| Multi-Fashion-MNIST | 1.00 | 0.8467 | 0.8462 | -0.0005 | 0.0739 |

All three candidates received `STOP`. Alpha `0.5` produced a small positive
Multi-MNIST gain but did not transfer to Multi-Fashion-MNIST. Correction ratio
scaled with alpha as designed, while validation gain did not. This is evidence
that making the correction task-aware and materially larger is not sufficient
to produce a general benchmark improvement.

These are two-seed exploratory results, not significance-tested paper results.
The gate prevents a five-seed test run of this formulation.

## Diagnostics

Each epoch records correction ratio, raw/corrected gradient alignment, mean
task conflict, conflict-active task fraction, minimum task/update alignment,
and held-out shared-trunk task-gradient cosine. The configuration records the
transport, alpha, conflict scope, and projection rule.

## Claim boundary

The rule is closely related to soft gradient surgery. PCGrad, CAGrad,
Nash-MTL, Gradient Vaccine, and strong scalarization baselines remain required
comparisons. Until those comparisons and a directional negative control are
complete, v2 is described as a task-aware Nash-inspired approximation rather
than a Nash-equilibrium solver or a new causal account of multi-task transfer.
