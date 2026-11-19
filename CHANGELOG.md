# Changelog

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
