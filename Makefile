# Reproduction targets. Runtimes measured on a 4-vCPU cloud VM (Xeon 2.3 GHz).
PY ?= .venv/bin/python
BH ?= .venv/bin/burhava

.PHONY: setup test lint reproduce-small reproduce-full stageA stageA-loss stageB reports

setup:
	uv venv --python 3.11 && uv pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

# ~1 minute (51 s measured): tests + a 20-graph-per-cell Stage A and a 10-graph Stage B, with reports.
# Numbers will be noisy; they check the pipeline, not the hypotheses.
reproduce-small:
	.venv/bin/pytest -q -m "not slow"
	$(BH) grid --config configs/stageA.yaml --out results/small/stageA --n-graphs 20
	$(BH) grid --config configs/stageB.yaml --out results/small/stageB --n-graphs 10
	$(BH) report --stage A --in results/small/stageA --figdir results/small/figures
	$(BH) report --stage B --in results/small/stageB --figdir results/small/figures

# Full pre-registered runs (see docs/EXPERIMENTS.md for measured runtimes).
reproduce-full: stageA stageA-loss stageB reports

stageA:
	$(BH) grid --config configs/stageA.yaml --out results/stageA --n-graphs 1000
stageA-loss:
	$(BH) grid --config configs/stageA_loss_sweep.yaml --out results/stageA_loss --n-graphs 1000
stageB:
	$(BH) grid --config configs/stageB.yaml --out results/stageB --n-graphs 500
reports:
	$(BH) report --stage A --in results/stageA
	$(BH) report --stage A-loss --in results/stageA_loss
	$(BH) report --stage B --in results/stageB
