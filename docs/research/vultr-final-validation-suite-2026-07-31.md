# Final validation suite - 2026-07-31

This is a pre-registered CPU queue for the remaining Vultr credit window. It
uses a clean source commit, a hard account-wide pending-charge limit of USD
`120`, and copies every ignored artifact back before deleting its worker.

## Evidence tiers

| Queue | Runs | What it can establish | What it cannot establish |
|---|---:|---|---|
| CIFAR-10 ResNet-18 locked study | 16 validation trials + 20 held-out test trials | Tuned, multi-seed image-classification comparison | ImageNet-scale quality or accelerator throughput |
| CIFAR-100 ResNet-34 locked study | 16 validation trials + 20 held-out test trials | A harder, 100-class vision transfer comparison | General ViT or ImageNet behavior |
| MNIST diffusion | 20 runs, five seeds, 50 epochs | Optimization stability on a generative loss | Sample quality; no FID claim without a sample-quality protocol |
| MuJoCo control | 60 runs, three environments, five seeds, 500k steps | Cross-environment optimizer stability and return variance | A claim of broad RL superiority |
| Core and backend gates | lint, tests, docs, Torch/JAX/TensorFlow smoke | Implementation and framework integration health | Optimizer quality |

The total planned result count is `132`, excluding reports and logs. The two
locked studies are the only planned headline-comparison artifacts because they
hold the CIFAR test set out of learning-rate selection.

## Dispatch order

1. Run lint, full tests, strict documentation build, and dataset checksum
   checks on the worker.
2. Run the CIFAR-10 locked study. Review the retained validation tuning table
   only after all four comparators have completed the same four-trial budget.
3. Run CIFAR-100 with the identical locked-study procedure.
4. Run diffusion and RL as separate evidence categories. Keep their results
   out of the vision headline table.
5. Generate `benchmarks.report --paper` only from held-out, clean-tree,
   complete-seed vision manifests. Report all other categories separately.

## Capacity status

On 2026-07-31, Vultr Cloud GPU creation remained blocked behind a support
request. The account also reached its monthly fee limit while two pre-existing
instances were active, so no additional benchmark worker could be created.
Those instances are not part of this suite and must not be repurposed or
deleted implicitly. The queue is ready to start as soon as capacity is made
available; no existing workload needs to be disturbed.

## CPU dispatch command

```bash
python -m benchmarks.torch_suite.locked_study \
    --study cifar10-resnet18 \
    --output-dir benchmark-runs/vultr-final-2026-07-31/cifar10-locked \
    --parallelism 4 --torch-threads 8 --workers 2
```

Run the same command with `--study cifar100-resnet34` for the second vision
study. A capacity worker must be cloned at the clean source commit before this
command starts; copying source files over an existing checkout makes its
manifests ineligible for strict paper reporting.
