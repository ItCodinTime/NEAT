# Reproduction guide

Install benchmark dependencies with `pip install -e ".[benchmarks]"`. Each run
manifest records the git revision and dirty state, package versions, hardware,
library seeds, exact command, histories, diagnostics, and wall time.

Generate paper artifacts from a complete run directory:

```bash
python -m benchmarks.report benchmark-runs/compare-all --paper
```

The report requires a `comparison_manifest.json` seed plan or an explicit
`--seeds` list. Paper mode rejects preliminary and dirty-tree runs. It emits
Markdown, LaTeX, full-precision JSON, accuracy curves with seed bands, and an
accuracy-versus-wall-time PDF.

After the author resolves the experiment settings, copy
`benchmarks/paper_experiments.example.json` to
`benchmarks/paper_experiments.json` and replace its commands with the approved
task configurations. Then run:

```bash
make reproduce-paper
make reproduce-paper FROM_MANIFESTS=1  # aggregation only
```

The command refuses to invent a paper plan when that author-owned file is
absent.

Dataset artifacts that are not managed by a framework loader can be installed
without an unchecked manual copy:

```bash
python scripts/fetch_dataset.py URL data/archive --sha256 EXPECTED_SHA256
```

The download is installed only after its required SHA-256 digest matches.
