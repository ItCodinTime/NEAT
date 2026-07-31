# Validation-locked benchmark protocol

This protocol is the publication-quality path for the PyTorch vision suite. It
prevents two common optimizer-evaluation errors: selecting a learning rate on
the test set, and comparing an extensively tuned method with an untouched
baseline.

## Procedure

1. Reserve a deterministic 10% slice of the original CIFAR training set for
   validation using `split_seed=20260731`. Training augmentation is applied
   only to the remaining 90%; validation and test transforms are deterministic.
2. Give each optimizer four documented learning-rate candidates. Candidate
   ranges may differ by optimizer family because their stable scales differ,
   but every family receives the same four-trial budget and the complete trial
   table is retained.
3. Select exactly one learning rate per optimizer from the final validation
   metric using tuning seed `101`. The selection is written to
   `tuning_manifest.json` before any test process starts.
4. Train fresh models with those locked settings for the pre-registered test
   seeds. The test set is used only for this phase.
5. Generate paired multi-seed statistics and retain the raw manifests, logs,
   locked selection file, commit hash, hardware, package versions, and NEAT
   diagnostics.

The executable implementation is
`python -m benchmarks.torch_suite.locked_study`. It runs the two phases in
order, labels tuning and test manifests distinctly, and invokes the strict
paper report only after all expected held-out runs exist.

## Scope and limitations

This is a controlled optimizer study, not an ImageNet-scale claim. It reports
Top-1 CIFAR test accuracy, convergence, wall time, and paired uncertainty. A
separate suite is required before making claims about large-scale vision,
foundation-model fine-tuning, diffusion sample quality, or RL generality.

The protocol does not tune architecture, augmentation, schedules, or training
duration per optimizer. Those choices remain fixed so it measures optimizer
behavior under one shared recipe. A future large-budget study may allocate a
separate, equal search budget to these additional recipe parameters, provided
the search plan is registered before viewing test results.
