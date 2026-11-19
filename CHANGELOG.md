# Changelog

## M6 — Write-up package (2026-11-19)
* `README.md` (summary, honest scope, quickstart, results table, figures, "What this is not"), `docs/RESULTS.md` (every hypothesis, numbers with CIs, sources).
* `paper/main.tex` skeleton; `scripts/make_paper_numbers.py` writes `paper/numbers/*.tex` from `results/summaries/`.
* `Makefile`: `reproduce-small` (51 s measured) and `reproduce-full` (about 45 min on 4 cores).

## M5 — Stage C: offline part only (2026-11-19)
* `llm/proofwriter.py`, `llm/prontoqa.py`: loaders, gold dependency graphs, controlled premise corruption, truth by forward chaining, dependency-edge F1.
* `llm/client.py` (call accounting per arm; the real client refuses without `BURHAVA_ALLOW_SPEND=1`), `llm/prompts.py`, offline tests with a mock LLM.
* `scripts/download_data.sh` (checksums), `scripts/estimate_stage_c_cost.py`.
* **Not run:** no API calls; H5 not evaluated. It is waiting for budget approval and one design decision (D23).

## M4 — Stage B: noise robustness (2026-11-19)
* `noise.py`: logit-space miscalibration (σ calibrated to target ECE by `scripts/calibrate_ece.py`), edge corruption (drop/rewire/add), verifier error. `stage_b.py`: paired runner and report.
* Ran 24 cells × 500 graphs × 72 noise conditions (8.6M episodes, 11 min).
* **Gate 2 passed:** H4 supported. At ECE 0.10 with verifier 5%, the gain is 84% of its clean value, with the CI above 0. Calibration error up to 0.20 does not remove the advantage.
* **Boundary found:** dependency-edge errors. Exploratory fine grid: the gain holds at ≤ 2% edge error, vanishes at 5%, and turns negative at 10%. All graph-based arms become overconfident.

## M3 — Stage A: oracle synthetic (2026-11-19)
* `PREREGISTRATION.md` frozen in its own commit before any Stage A data existed.
* `generators.py`: two independent generators (layered AND/OR DAGs; ProofWriter-style rules with distractors), plus structure statistics.
* `runner.py` / `cli.py` (`burhava run|grid|report`), `stats.py` (paired bootstrap, ECE), `analysis.py` (hypothesis tests, tables, figures), `noise.independent_version`.
* Ran 256 cells × 1,000 graphs (11.8M episodes, 20 min) and a loss sweep.
* **Gate 1 passed:** H1 supported in all 4 structures; H2 supported (small effect); H3 rejected (ignoring correlation costs 4.1% of the gain; the threshold was 5%).
* Exploratory finding: the loss-blind propagation score (arm 8) matches arm 10 at budgets ≤ 2 and beats it at budgets ≥ 3, because arm 10 stops early.

## M2 — Policies and simulator (2026-11-19)
* `policies.py`: the 10 arms with one interface `select(state, budget_left)`. Ties go to lower cost, then lower index.
* `simulate.py`: paired worlds (true state, every verdict and every SC sample drawn before any arm runs). Each episode returns decision, true Y, realised loss, model risk at stop, P(Y), checks, cost, risk evaluations and wall time.
* `risk.py`: batched `single_risks` and `pair_risks`. `model.py`: noisy conclusion readings (`observe_y`).
* Tests: a unit test per arm; arm 9 stalls on the AND counterexample and arm 10 does not (it reaches zero loss with budget 2); arms 9 and 10 pick Z1 on the selection example while arms 3 and 4 pick Z3.

## M1 — Exact engine (2026-11-19)
* `graph.py`: AND/OR assumption graphs with step-validity atoms, negated premises, shared sources, per-atom cost and verifier errors.
* `model.py`: exact joint `P(z)` under the shared-source model, noisy verifier channel, Bayes update, posterior marginals.
* `risk.py`: loss matrices, ρ, R(S) (fast exact via binning + channels), Δ.
* Tests: the 4 acceptance tests (AND, XOR, selection example, 500-case monotonicity property) plus unit tests; fast R(S) cross-checked against the definition.
* Timing: R(S) at n=20 takes 4–21 ms (|S| ≤ 5).
* Docs: RESEARCH_BRIEF, MATH (step-by-step derivations), DECISIONS (D1–D10), RELATED_WORK, EXPERIMENTS. CI: ruff + fast pytest on Python 3.11.
