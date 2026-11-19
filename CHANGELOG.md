# Changelog

## M1 — Exact engine (2026-11-19)
* `graph.py`: AND/OR assumption graphs with step-validity atoms, negated premises, shared sources, per-atom cost and verifier errors.
* `model.py`: exact joint `P(z)` under the shared-source model, noisy verifier channel, Bayes update, posterior marginals.
* `risk.py`: loss matrices, ρ, R(S) (fast exact via binning + channels), Δ.
* Tests: the 4 acceptance tests (AND, XOR, selection example, 500-case monotonicity property) plus unit tests; fast R(S) cross-checked against the definition.
* Timing: R(S) at n=20 takes 4–21 ms (|S| ≤ 5).
* Docs: RESEARCH_BRIEF, MATH (step-by-step derivations), DECISIONS (D1–D10), RELATED_WORK, EXPERIMENTS. CI: ruff + fast pytest on Python 3.11.
