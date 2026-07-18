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

# 6. Generate exact tables, paired tests, cost summaries, and plots.
python -m benchmarks.sts.analyze benchmark-runs/sts

# 7. Capture the final machine, packages, git state, and dataset hashes.
python -m benchmarks.sts.environment
```

Every trial writes `config.json`, append-only `metrics.jsonl`, and a final
`result.json`. Trial paths are deterministic. Re-running a completed command
skips it unless `--overwrite` is explicit, so interrupted matrices can resume
without replacing completed evidence.

The full protocol and claim boundaries are in
`docs/research/regeneron-sts-protocol.md`.
