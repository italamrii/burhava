# Research brief

## Question
An LLM reasoning chain rests on assumptions and inference steps, any of which may be wrong. We can afford to verify only a few. **Which should we check?**

The selection rule we test is classical **Value of Information (VoI)**: check the atom whose verdict is expected to reduce *decision loss* the most per unit cost. We do **not** claim this rule is new. The contribution under test is *empirical*:

1. Does decision-risk-aware selection beat confidence-based selection (for example, "check the least confident premise first") for LLM reasoning?
2. Does the advantage survive when the model's inputs are noisy: miscalibrated probabilities, wrongly recovered dependency edges, and verifiers whose error rates are misestimated?

## Model
* Atoms `Z_i ∈ {0,1}`: assumptions plus step-validity variables. The conclusion is `Y = f(Z)`, built from AND/OR rules. A derived node is true if any alternative derivation holds. A derivation holds if all its premises hold and its step is valid.
* Verifying atom *i* costs `c_i` and returns a verdict `O_i` that may be wrong, with false-positive rate `α_i = P(O_i=1∣Z_i=0)` and false-negative rate `β_i = P(O_i=0∣Z_i=1)`.
* The final decision is `d ∈ {accept, reject, abstain}`, with loss `L(d, y)`.
* Bayes risk `ρ(I) = min_d E[L(d,Y)∣I]`. Set risk `R(S) = E_{O_S}[ρ(E, O_S)]`. Marginal value `Δ_i(S) = R(S) − R(S∪{i})`.
* Correlation: each assumption cites a source `s(i)` with latent reliability `U_k`, and `P(Z) = Σ_u P(u) Π_i P(Z_i ∣ U_{s(i)})`.

Full derivations are in [MATH.md](MATH.md).

## Known facts the engine must reproduce (acceptance tests, `tests/test_acceptance.py`)
| Fact | Setup | Expected | Status |
|---|---|---|---|
| AND counterexample | `Y=Z1∧Z2`, p=.5, perfect verifier, 0–1 loss | R(∅)=R({1})=R({2})=.25, R({1,2})=0 | ✅ exact |
| XOR | same, `Y=Z1⊕Z2` | .5, .5, .5, 0 | ✅ exact |
| Selection example | `Y=Z1∧(Z2∨Z3)`, p=(.9,.6,.55), losses 10/1/.6 | R(∅)=.6, Δ=(.060,.042,.027) | ✅ exact |
| Monotonicity | 500 random graphs, losses, verifier errors | R(S∪{i}) ≤ R(S) | ✅ 500/500 |

## Plan (six milestones)
M1 exact engine → M2 ten selection policies and simulator → M3 Stage A: oracle synthetic, with pre-registration frozen at M3 start (Gate 1) → M4 Stage B: noise robustness (Gate 2) → M5 Stage C: real LLMs on ProofWriter/PrOntoQA (API spend needs approval) → M6 write-up.

The hypotheses H1–H5 and their rejection criteria go into `PREREGISTRATION.md`, verbatim, at the start of M3.

## Prior art
See [RELATED_WORK.md](RELATED_WORK.md).
