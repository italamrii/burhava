# Pre-registration (frozen 2026-11-19, at the start of M3, before any Stage A data was generated)

This file must not be edited after its first commit. Any later deviation goes into `docs/DECISIONS.md` and is reported as a deviation.

## Hypotheses (verbatim from the project brief)

| ID | Hypothesis | Rejected if |
|---|---|---|
| H1 | Stage A: proposed (arm 10) lowers mean decision loss vs lowest-confidence-first (arm 3) | Paired 95% bootstrap CI of the difference includes 0 in ≥ 3 of 4 structures at budget 2 |
| H2 | Lookahead matters on AND | No significant gain of arm 10 over arm 9 on pure-AND graphs at budgets 2–3 |
| H3 | Modelling correlation matters | Shared-source vs independence variants differ by < 5% of the H1 gain in correlated cells |
| H4 | Advantage survives realistic noise | At ECE 0.10 and verifier error 5%, gain over arm 3 < 25% of its Stage A value, or CI includes 0 |
| H5 | Advantage transfers to benchmarks | Stage C CI of gain over best of arms 3–5 includes 0 |

Metrics: primary = mean realised decision loss. Secondary = error rate, risk reduction per unit cost, Brier, ECE of final P(Y), dependency-recovery F1 (Stage C), runtime, risk evaluations. Bootstrap: 10,000 resamples, paired by instance.

## Operational definitions (written before data; they make the table above computable)

**General**
* **Instance:** one generated graph with one pre-drawn world (true state + all verdicts). Every arm runs on the same instance (paired).
* **Gain of arm A over arm B:** `mean_instances[ loss(B) − loss(A) ]`, using realised loss. Positive means A is better. CIs are percentile intervals from 10,000 paired bootstrap resamples of instances, with seed 0.
* **Main loss:** false-accept 10, false-reject 1, abstain 0.6, correct decisions 0. The loss sweep (ratios 1, 3, 10, 30 of false-accept to false-reject, keeping false-reject 1 and abstain 0.6) is secondary and descriptive.
* **Arm parameters:** arm 10 with η = 0.05 (relative; D15). Arm 6 with ε_SC = 0.3. Not tuned.
* **Stage A world:** oracle model (the method's model equals the truth), perfect verifier. The factors are those of the brief's grid, with the encodings in `src/burhava/generators.py`. Each cell's graphs are split evenly between the two generators. H tests pool both generators; per-generator results are reported as well.
* **Sample size:** 1,000 graphs per cell are planned. If the measured runtime makes this infeasible, a smaller N is chosen **from timing alone, before looking at any outcome**, and is reported.

**H1:** For each structure ∈ {chain, and, or, mixed}, at budget 2 under the main loss, pooled over depth, false-rate, correlation, cost and generator: compute the gain of arm 10 over arm 3 and its 95% CI. A structure *fails* if its CI includes 0. A CI entirely below 0 (arm 10 worse) also counts as a failure; this is stricter than the brief. **H1 is rejected if ≥ 3 of 4 structures fail.**

**H2:** Structure = "and" (both generators), budgets 2 and 3 pooled, main loss. **Rejected unless** the 95% CI of the gain of arm 10 over arm 9 lies entirely above 0.

**H3:** Correlated cells (correlation = shared), all budgets pooled, main loss. Arm "10-indep" is arm 10 run with an independence model (each atom keeps its true marginal; sources are dropped). Its beliefs and decision use that model, while the world stays correlated. Let `D = mean[loss(10-indep) − loss(10)]` and `G = mean[loss(3) − loss(10)]` on the same instances. **Rejected if |D| < 0.05·G.** If G ≤ 0, H3 is reported as not evaluable.

**H4 (Stage B):** M4 runs a sub-grid: all 4 structures, depth 4, false-rate {0.1, 0.2, 0.3}, both correlations, uniform cost, budget 2, main loss, 500 graphs per cell. Within it:
* `G_A` = the gain of arm 10 over arm 3 with no noise.
* `G_B` = the same gain at target ECE 0.10 with true verifier error 5%, where the method assumes the correct 5% ("matched") and edge error is 0.

**Rejected if G_B < 0.25·G_A, or the 95% CI of G_B includes 0.** Mismatched-verifier and edge-error conditions are secondary. If H4 is rejected, the calibration threshold X (the largest ECE at which the gain's CI still excludes 0) is estimated on the ECE grid {0, .05, .10, .20}.

**H5 (Stage C):** the best of arms 3–5 is the one with the lowest mean loss on the Stage C test split, chosen after the fact, which favours the baseline. **Rejected if the 95% paired CI of the gain of arm 10 over that arm includes 0.** Every LLM call is counted per arm.

**Gates:** Gate 1 = H1. If H1 is rejected: stop, write `docs/NEGATIVE_RESULT.md`, report. Gate 2 = H4. If H4 is rejected, the paper becomes a boundary result; continue to M5 either way.
