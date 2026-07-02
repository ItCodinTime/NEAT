# Baseline tuning protocol

`benchmarks.tuning.run_tuning` expands one documented learning-rate and
weight-decay grid for every optimizer on a task. It records every validation
result in `tuning_manifest.json`, writes `tuning_report.md`, and selects one
configuration per optimizer using only the configured validation direction.

`require_all_optimizers_tuned` rejects comparisons unless the tuned optimizer
set exactly equals the compared optimizer set. PCGrad and CAGrad are only
applicable where separate task or player gradients exist; they are not assigned
a synthetic single-task momentum-opponent analogue.

