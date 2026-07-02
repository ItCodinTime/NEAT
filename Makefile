.PHONY: reproduce-paper

RUNS_DIR ?= benchmark-runs/paper
ASSETS_DIR ?= paper_assets

reproduce-paper:
	python scripts/reproduce_paper.py $(if $(FROM_MANIFESTS),--from-manifests,) --runs-dir $(RUNS_DIR) --assets-dir $(ASSETS_DIR)
