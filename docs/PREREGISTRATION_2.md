# Pre-registration 2: fixing the two weaknesses (frozen 2026-11-19, before any phase-2 data)

This file must not be edited after its first commit. Phase 1 (`PREREGISTRATION.md`) found two weaknesses in exploratory analyses. Phase 2 tests one proposed fix for each, on **fresh seeds** (20261201, 20261202) that were never used before. Nothing about the arms below was tuned on phase-2 data; the parameters were fixed while writing this file.

## Arms
* **Arm 11, "VoI then spend"** (`policies.VoIThenSpend`): behaves as arm 10 (η = 0.05). When arm 10 would stop with budget left, it spends the rest by propagation score (arm 8).
* **Arm 12, "robust VoI"** (`model.MixtureProblem`, `noise.graph_ensemble`): arm 10's selection rule, run on a belief that averages over K = 8 candidate graphs, each with weight 1/8. The candidates are the recovered graph plus 7 copies whose edges are re-corrupted at an assumed rate r̂ = 0.10. The method does **not** know the true edge-error rate; r̂ is fixed at 0.10 in every condition.

## Experiments
* **P2-A** (`configs/phase2_A.yaml`): an oracle world like Stage A.
  * Grid: structure {chain, and, or, mixed} × depth {4, 8} × false-rate {.1, .2, .3} × correlation {independent, shared} × cost {uniform, varied}.
  * Budgets {1, 2, 3, 5}; main loss (10 / 1 / 0.6); 1,000 graphs per cell.
  * Arms 3, 8, 10, 11.
* **P2-B** (`configs/phase2_B.yaml`): the Stage B sub-grid (depth 4, uniform cost, budget 2, main loss).
  * Verifier error 5%, matched. ECE {0, 0.10}. True edge error {0, 0.05, 0.10, 0.25}.
  * 1,000 graphs per cell. Arms 1, 3, 6, 8, 10, 12.

Gains are `mean[loss(B) − loss(A)]` per instance, with 95% percentile CIs from 10,000 paired bootstrap resamples (seed 0). Positive means arm A is better.

## Hypotheses and rejection criteria
| ID | Hypothesis | Rejected if |
|---|---|---|
| H6a | Arm 11 beats arm 10 at budgets 3 and 5 (pooled, P2-A) | CI of gain(11 over 10) does not lie entirely above 0 |
| H6b | Arm 11 is non-inferior to arm 8 at all budgets (pooled, P2-A) | CI lower bound of gain(11 over 8) < −0.002 |
| H6c | Decision-awareness adds value: arm 11 beats arm 8 (all budgets pooled, P2-A) | CI of gain(11 over 8) does not lie entirely above 0 |
| H7a | Robust VoI beats VoI under 10% edge error (P2-B, ECE 0.10) | CI of gain(12 over 10) does not lie entirely above 0 |
| H7b | Robust VoI restores the advantage over arm 3 at 10% edge error (P2-B, ECE 0.10) | CI of gain(12 over 3) does not lie entirely above 0 |
| H7c | Robustness is cheap when the graph is right (P2-B, edge 0, ECE 0.10) | mean[loss(12) − loss(10)] > 0.0162 (= 25% of the phase-1 H4 gain, 0.0648) |
| H7d | Robust VoI is no longer harmful at 25% edge error (P2-B, ECE 0.10) | CI of gain(12 over arm 1, no verification) does not lie entirely above 0 |

All seven are reported whatever the outcome. Secondary analyses (per structure, the ECE 0 condition, arm 6) are descriptive.
