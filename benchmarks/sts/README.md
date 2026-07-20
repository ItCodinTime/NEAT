# Regeneron STS experiment suite

This package runs the paper's controlled Multi-MNIST and Multi-Fashion-MNIST
study. It keeps tuning, final test evaluation, mechanism evidence, ablations,
and reporting as separate resumable phases.

## Order of operations

```bash
# 1. Fast integration gate. These results are always preliminary.
python -m benchmarks.sts.run_suite smoke --seeds 7

# 2. Select one learning rate per dataset and method using validation only.
python -m benchmarks.sts.tune

# 3. Run the locked five-seed core matrix.
python -m benchmarks.sts.run_suite tier1

# 4. Add matched single-task NEAT runs for mechanism evidence.
python -m benchmarks.sts.run_suite mechanism

# 5. Run one-factor NEAT ablations.
python -m benchmarks.sts.run_suite ablations

# 6. Run the validation-only task-aware v2 promotion gate.
python -m benchmarks.sts.run_suite promotion --seeds 101 103 --epochs 5
python -m benchmarks.sts.analyze_promotion benchmark-runs/sts-v2-gate

# 7. Explore whether measured task conflict changes with image overlap.
python -m benchmarks.sts.run_suite severity \
  --seeds 101 103 107 --epochs 5
python -m benchmarks.sts.analyze_severity benchmark-runs/sts-conflict-severity

# 8. Generate exact tables, paired tests, cost summaries, and plots.
python -m benchmarks.sts.analyze benchmark-runs/sts

# 9. Capture the final machine, packages, git state, and dataset hashes.
python -m benchmarks.sts.environment
```

Every trial writes `config.json`, append-only `metrics.jsonl`, and a final
`result.json`. Trial paths are deterministic. Re-running a completed command
skips it unless `--overwrite` is explicit, so interrupted matrices can resume
without replacing completed evidence.

The `promotion` phase is exploratory and excluded from the original Tier 1
aggregation. It compares task-aware soft projection at `alpha=0`, `0.25`,
`0.5`, and `1.0` under the same Adam transport. The equation and stopping rule
are documented in `docs/research/task-aware-neat-v2.md`.

The full protocol and claim boundaries are in
`docs/research/regeneron-sts-protocol.md`.

The severity phase is exploratory and validation-only. It varies the diagonal
offset of the two overlaid images across `0`, `4`, and `8` pixels, records the
measured task-gradient cosine, and compares task-aware alpha `0.5` with its
exact alpha-zero Adam transport control. Overlap is treated as an intervention,
not as a substitute for measured conflict.
