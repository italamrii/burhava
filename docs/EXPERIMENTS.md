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
