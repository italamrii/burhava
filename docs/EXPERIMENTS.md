# Experiment log

One entry per run: date, git hash, command, config, runtime, key numbers, verdict.

## M1-timing — exact engine speed at n = 20 (2026-11-19)
* **Git:** the commit titled `M1: exact engine (graph, model, risk) + acceptance tests + docs`. The benchmark code is unchanged in that commit.
* **Command:** `python scripts/bench_m1.py --seed 0 --out results/summaries/m1_timing.json`
* **Config:** 20 atoms (14 assumptions, 9 of them across 3 binary-latent sources; 6 step atoms), alternating AND/OR layers, α=β=0.05, loss (false-accept 10, false-reject 1, abstain 0.6). Medians of 7 repeats (3 for the Δ scan).
* **Machine:** cloud container, Intel Xeon @ 2.30 GHz, 4 vCPUs, Python 3.11.15, numpy 2.4.6.
* **Results:**

| Operation | Median time |
|---|---|
| Build joint table + Y (`Problem(g)`) | 0.140 s (once per instance) |
| ρ (Bayes risk) | 0.9 ms |
| R(S), \|S\| = 0 / 1 / 2 / 3 / 4 / 5 | 4.1 / 11.3 / 11.9 / 14.7 / 17.2 / 20.6 ms |
| Δ_i(S) for all 18 remaining atoms, \|S\| = 2 | 0.39 s |
| Bayes update (one verdict) | 2.0 ms |

* **Verdict:** ✅ The M1 criterion "n=20 risk evaluation < 1 s" is met with about 50× headroom. The slow test `tests/test_timing.py` guards it.
* **Note:** These are laptop-class numbers from a shared cloud VM. Expect ±30%.

## M3-pilot — Stage A runtime pilot, 100 graphs/cell (2026-11-19)
* **Git:** run on working tree after `fb7eedc` (pre-registration commit); the code is committed unchanged in the M3 commit.
* **Command:** `burhava grid --config configs/stageA.yaml --out results/stageA_pilot100 --n-graphs 100 --workers 4`
* **Config hash:** 8b1d5e3d2f. **Runtime:** 115 s.
* **Use:** runtime projection only (→ about 19 min for N=1000), plus debugging of the analysis code. Not used for any decision about the design (D20).

## M3-A — Stage A main grid, 1000 graphs/cell (2026-11-19)
* **Git:** working tree after `fb7eedc`; the code is committed unchanged in the M3 commit.
* **Command:** `burhava grid --config configs/stageA.yaml --out results/stageA --n-graphs 1000 --workers 4`, then `burhava report --stage A --in results/stageA`
* **Config hash:** 8b1d5e3d2f. **Runtime:** 1,192 s (grid) and 530 s (report).
* **Size:** 256 cells × 1,000 graphs × 4 budgets × 10 arms, plus arm 10-indep in shared cells. 11.8M episodes.
* **Key numbers:** budget 2, main loss. The gain is `loss(arm 3) − loss(arm 10)`, with 95% paired bootstrap CIs and n = 64,000 per structure.

| Structure | Gain | 95% CI |
|---|---|---|
| chain | +0.0298 | [+0.0277, +0.0319] |
| and | +0.0028 | [+0.0022, +0.0035] |
| or | +0.0824 | [+0.0778, +0.0869] |
| mixed | +0.1116 | [+0.1068, +0.1162] |

* **H2** (and, budgets 2–3, arm 10 vs arm 9): +0.0041 [+0.0036, +0.0046].
* **H3** (shared cells, all budgets): D = loss(10-indep) − loss(10) = +0.0020 [+0.0016, +0.0025]. G = +0.0496. |D|/G = 4.1%.
* **Verdicts:** H1 supported (0 of 4 structures fail); **Gate 1 passed**. H2 supported. **H3 rejected** (4.1% < 5%).
* **Exploratory (not pre-registered):** arm 8 (propagation score) matches arm 10 at budgets 1–2 and beats it at budgets 3–5. For example, chain at budget 5: arm 10 − arm 8 loss = +0.0194 [+0.0182, +0.0206]. Arm 10 stops once no single check or pair has positive value; arm 8 spends the whole budget.

## M3-L — Stage A loss sweep (2026-11-19)
* **Command:** `burhava grid --config configs/stageA_loss_sweep.yaml --out results/stageA_loss --n-graphs 1000 --workers 4`, then `burhava report --stage A-loss --in results/stageA_loss`
* **Config hash:** b2c6c802da. **Runtime:** 200 s.
* **Key numbers:** the gain over arm 3 rises with the false-accept/false-reject ratio in OR and mixed graphs (mixed: +0.079 at ratio 1 → +0.193 at ratio 30). It is flat in chain (+0.03 to +0.06) and small in AND (+0.003 to +0.007). Full table: `results/summaries/stageA_loss_summary.json`.

## M4-B — Stage B noise robustness, 500 graphs/cell (2026-11-19)
* **Git:** working tree after `7d3b4fc` (M3); the code is committed unchanged in the M4 commit.
* **Calibration of σ:** `python scripts/calibrate_ece.py --seed 999` gave σ = 0.637, 1.032 and 1.989.
* **Command:** `burhava grid --config configs/stageB.yaml --out results/stageB --n-graphs 500 --workers 3`, then `burhava report --stage B --in results/stageB`
* **Config hash:** 77a04bbd1e. **Runtime:** 688 s (grid) and 93 s (report).
* **Size:** 24 cells × 500 graphs × 72 noise conditions × 10 arms. 8.64M episodes.
* **Achieved noise:** population ECE was 0.046, 0.092 and 0.189 for targets 0.05, 0.10 and 0.20. Edge error was 0.100 and 0.250 for targets 0.10 and 0.25.
* **H4:** G_A (clean) = +0.0775 [+0.0701, +0.0852]. G_B (ECE 0.10, verifier 5% matched) = +0.0648 [+0.0522, +0.0775]. G_B/G_A = 0.84 ≥ 0.25 and the CI excludes 0, so **H4 is supported and Gate 2 passed**. By structure at the H4 point: chain +0.035, **and +0.001 [−0.001, +0.004] (n.s.)**, or +0.114, mixed +0.109.
* **Calibration:** the gain stays positive at every ECE level tested (up to 0.20) and every verifier error (0/5/15%, matched or mismatched). The one exception is verifier 15% mismatched at ECE 0.05: +0.014 [−0.001, +0.029].
* **Edge errors (secondary, pre-registered levels):** at 10% edge error the gain over arm 3 turns negative (for example −0.045 [−0.066, −0.025] at verifier 5%, ECE 0). At 25% it is strongly negative (−0.13 to −0.21). Arms 8, 9 and 10 all fall below no-verification at 25% edge error: their model risk is about 0.20 while the realised loss is about 0.53, i.e. overconfidence.

## M4-E — EXPLORATORY (not pre-registered): fine edge-error grid (2026-11-19)
* **Command:** `burhava grid --config configs/stageB_edge_explore.yaml --out results/stageB_edge --n-graphs 500`, then `python scripts/explore_edge.py results/stageB_edge`
* **Config hash:** e65010a80c. **Runtime:** 63 s. Seed 20261122, independent of the Stage B seed.
* **Gain of arm 10 over arm 3** (verifier 5% matched):

| Edge error | ECE 0 | ECE 0.10 |
|---|---|---|
| 0 | +0.052 [+0.040, +0.063] | +0.065 [+0.051, +0.080] |
| 2% | +0.034 [+0.020, +0.048] | +0.049 [+0.033, +0.065] |
| 5% | −0.005 [−0.023, +0.013] | +0.013 [−0.008, +0.033] |
| 10% | −0.048 [−0.070, −0.028] | −0.023 [−0.046, −0.001] |

* **Reading:** the advantage needs dependency-edge error ≲ 2–5%. At 10%, arm 10 is worse than arm 3 and worse than not verifying at all.

## M5 — Stage C: NOT RUN (awaiting API budget approval) (2026-11-19)
* Data downloaded and checksummed (`scripts/download_data.sh`). 2,900 usable ProofWriter instances (mean 5.4 atoms) and 1,400 PrOntoQA instances.
* Cost estimate: `python scripts/estimate_stage_c_cost.py --n-test 600 --n-cal 200 --seed 0` gives 62,163 calls, ~11.87M input tokens and ~4.04M output tokens.
* H5: not evaluated.
