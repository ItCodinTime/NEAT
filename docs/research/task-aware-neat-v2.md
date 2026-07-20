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
