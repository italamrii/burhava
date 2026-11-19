# Decision log (ADR style)

Each entry records the date, the decision, the alternatives considered and the reason.

## D1 — Exact enumeration over all 2^n states, capped at n ≤ 20 (2026-11-19, M1)
* **Decision:** Compute `P(z)`, `Y` and every risk by enumerating all `2^n` joint states with vectorised numpy. Cap `n` at 20 (`graph.MAX_ATOMS`).
* **Alternatives:** variable elimination or knowledge compilation (for example ProbLog/d-DNNF); Monte-Carlo estimation.
* **Reason:** The acceptance tests need exact numbers, and exactness removes estimator noise from the Stage A comparisons. At n=20, an R(S) evaluation takes ≤ 21 ms (EXPERIMENTS.md M1-timing). Memory is ~1M states × (20 bits + a few float vectors) ≈ 60 MB.

## D2 — Step-validity variables are ordinary atoms (2026-11-19, M1)
* **Decision:** "This step is valid" is an atom of kind `step`, with its own prior, cost, verifier errors and source. It counts toward the n ≤ 20 cap.
* **Alternatives:** a separate step-validity type with fixed probability; folding step validity into premise probabilities.
* **Reason:** The brief lets steps be verified and be wrong. A single atom type keeps `R(S)` and all policies uniform. M3 generators must budget atoms accordingly. A step known to be valid is written `step=None`.

## D3 — Negated premises are supported (2026-11-19, M1)
* **Decision:** A premise can be `neg("Z")`.
* **Alternatives:** monotone AND/OR only, with XOR tested through a separate arbitrary-function hook.
* **Reason:** The XOR acceptance test needs non-monotone `f`. ProofWriter's closed-world negation (Stage C) also needs it. Monotonicity of `R` (MATH §5.4) does not depend on `f` being monotone.

## D4 — "0–1 loss" sets abstain loss to 1 (2026-11-19, M1)
* **Decision:** `Loss.zero_one()` uses `L(abstain, ·) = 1`, so abstaining is never strictly better than guessing.
* **Alternatives:** abstain = 0.5, or a 2-decision problem.
* **Reason:** The brief's AND/XOR numbers (R(∅)=0.25 and 0.5) assume a binary guess. Any abstain loss ≥ 0.5 gives the same numbers; 1 makes the intent explicit.

## D5 — Each atom is verified at most once (2026-11-19, M1)
* **Decision:** `Belief.observe` and `set_risk` reject atoms that are already verified or appear twice.
* **Alternatives:** allow repeated checks, with verdicts independent given `Z_i`.
* **Reason:** Repeated LLM checks of the same claim have strongly correlated errors. Treating them as independent would overstate their value. This can be revisited in Stage C with an explicit repeat-correlation model.

## D6 — Source model: discrete latent, independent sources, at most one source per atom (2026-11-19, M1)
* **Decision:** `U_k` takes finitely many values. Sources are mutually independent. An atom cites zero or one source.
* **Alternatives:** continuous `U_k` (Beta); atoms citing several sources.
* **Reason:** This is exactly the brief's formula, and the joint stays exact and cheap (it factorises per source). A Beta latent can be approximated with a grid of values.

## D7 — Unnormalised posterior, and the assumed model kept separate from the true world (2026-11-19, M1)
* **Decision:** `Belief` stores `w(z) = P(z)·likelihood` and the `Problem` the *method* believes in. The true world (used in M2+ to sample `Z` and verdicts) is a separate `Problem`.
* **Alternatives:** normalise after every update; one shared problem.
* **Reason:** Skipping normalisation saves a pass and keeps `W = P(evidence)` available. Keeping the two problems separate is what Stage B needs: the method sees corrupted inputs while ground truth stays clean.

## D8 — Decision ties go to the lower index (accept < reject < abstain) (2026-11-19, M1)
* **Reason:** This is deterministic, and the risk value itself is unaffected by ties. In the AND example (P(Y)=.25) there is no tie. In XOR (P(Y)=.5), the engine accepts.

## D9 — Python 3.11 via uv; hatchling build; CI on 3.11 (2026-11-19, M1)
* **Reason:** This follows the brief. The container's default Python is 3.13, so 3.11.15 was installed with `uv python install 3.11` and all tests run on it.

## D10 — Fast R(S) by binning on z_S and applying 2×2 channels (2026-11-19, M1)
* **Decision:** `R(S)` is computed with one `bincount` over 2^n states into 2^|S| bins, followed by per-atom 2×2 channel contractions, at cost `O(2^n + |S|2^|S|)`.
* **Alternatives:** enumerate `o_S` and Bayes-update the full `w` each time, at cost `O(2^|S| · |S| · 2^n)`.
* **Reason:** It is 10–30× faster at |S| = 3–5. It is checked against the definition-level version on 200 random graphs (`test_fast_matches_naive`).

## D11 — Budget is measured in cost units (2026-11-19, M2)
* **Decision:** `budget` is the total verification cost allowed. Under uniform cost (`c_i = 1`), it is the number of checks. Under varied cost, costs are integers 1–5 rescaled to mean 1 per instance, so a budget of B buys about B average checks.
* **Alternatives:** budget = number of checks regardless of cost.
* **Reason:** The brief has a cost factor {uniform, 1–5}. Only a cost budget lets cost matter.

## D12 — Heuristic arms ignore cost and use the full budget (2026-11-19, M2)
* **Decision:** Arms 2–5 and 8 score atoms without dividing by cost and stop only when nothing is affordable. Arm 7 also stops at the first failure. Arms 9 and 10 stop when no check has positive value.
* **Reason:** This is how confidence-based selection is used in practice. The cost factor in the grid shows how much of any gain comes from cost-awareness alone (uniform-cost cells contain no such advantage).

## D13 — Self-consistency arm model (2026-11-19, M2)
* **Decision:** Arm 6 draws samples of the *conclusion*. Each costs 1 and is wrong with probability ε_SC = 0.3, independently given Y. The final decision uses the exact Bayes posterior.
* **Alternatives:** samples drawn from the model's own belief (which would carry no information about the truth); correlated samples.
* **Reason:** Independent samples with a Bayes combination are *generous* to this baseline. ε_SC is a config parameter. The real arm-6 behaviour is measured in Stage C.

## D14 — Arm 8 "propagation score" definition (2026-11-19, M2)
* **Decision:** `score_i = 2 p_i (1−p_i) |P(Y=1|Z_i=1,E) − P(Y=1|Z_i=0,E)|`. This is the expected absolute change in P(Y=1) if Z_i were revealed.
* **Reason:** The brief does not define the score. This one uses the graph and the probabilities but neither the loss nor the costs, so it isolates "decision-awareness" from "structure-awareness".

## D15 — Arm 10 lookahead trigger η = 0.05 (relative) (2026-11-19, M2)
* **Decision:** Arm 10 runs the two-step lookahead when `max_i Δ_i/c_i ≤ η·ρ(E)`, with η = 0.05. It then takes the better of the best pair rate and the best single rate. From the best pair it verifies the member with the higher single score, and it re-plans after every verdict.
* **Alternatives:** an absolute η; always look ahead (η = ∞); only when stalled (η = 0).
* **Reason:** A relative threshold is unit-free across loss scales. This value is fixed before Stage A and frozen in PREREGISTRATION.md. It is not tuned.

## D16 — Arm 7 order and verdicts impossible under the model (2026-11-19, M2)
* **Decision:** Arm 7's "topological order" is the order in which atoms are first used when scanning derived nodes in definition order (premises, then step). If a verdict has probability 0 under a misspecified model, the atom is counted as checked, no update is made, and a `contradictions` counter is incremented.
* **Reason:** Determinism. The second case cannot occur in Stage A (priors are in (0,1)) but can in Stage B.

## D17 — Batched one-step and pair risks (2026-11-19, M2)
* **Decision:** `single_risks` and `pair_risks` compute R({i}) for all i and R({i,j}) for all pairs, using matrix products of the bit matrix with the posterior. They are cross-checked against `set_risk` in tests.
* **Reason:** Arms 9 and 10 evaluate these at every step. Vectorising makes Stage A affordable.

## D18 — Generator encodings of the grid factors (2026-11-19, M3, before data)
* **Decision:** See the `generators.py` docstring.
  * The false-rate is the marginal prior false probability `f·g_i`, with `g_i ~ U(0.5, 1.5)`.
  * Shared-source correlation uses 2–3 sources, `P(bad) = 0.3`. A bad source makes its atoms 4× as likely to be false. Marginals are kept equal to the independent case (verified: mean false probability 0.15 under both).
  * Atoms are capped at 14.
  * Stage A uses a perfect verifier.
* **Known limitation, recorded before any arm was run:** under generator A, pure-OR graphs have P(Y)≈1.00 (many alternative routes), so verification can barely change the decision there. This was not tuned away.

## D19 — Two generators, alternating within each cell (2026-11-19, M3)
* **Decision:** Even graph ids come from generator A (layered), odd from generator B (rules). Hypothesis tests pool both; per-generator gains are reported. Generator B adds 1–2 distractor facts that are irrelevant to the conclusion (as in ProofWriter); generator A does not.

## D20 — Stage A N = 1000 graphs per cell (2026-11-19, M3)
* **Decision:** The 100-graph pilot took 115 s on 4 cores, projecting about 19 minutes for N = 1000. N = 1000 (as planned) was therefore run. The choice depended only on runtime. The pilot outcomes were not used for any decision (the pilot directory exists only to debug the analysis code).

## D21 — Stage C scope (2026-11-19, M5, offline part)
* **ProofWriter:** CWA depth-5 test; only negation-free theories (725/985), and only true questions with strategy "proof" and QDep ≥ 2, with no NAF. This gives 2,900 instances (mean 5.4 atoms, max 14).
* **PrOntoQA-OOD:** the "ProofsOnly" files; 1,400 usable items, each with a linear chain.
* **Corruption:** each context fact is replaced by its negated sentence with probability ρ. Truth is recomputed by forward chaining on the clean-minus-corrupted facts.
* **Atoms:** facts and rules used by the gold proofs; rules are step atoms.
* **Dependency recovery:** edges (premise id or producing rule) → rule.

## D22 — Stage B noise encodings (2026-11-19, M4)
* **Calibration:** `logit p' = logit p + σ·ε_i`, `ε_i ~ N(0,1)`, drawn once per graph and reused across σ (paired). σ per target comes from `scripts/calibrate_ece.py --seed 999` on a separate sample: 0.637, 1.032 and 1.989 for ECE 0.05, 0.10 and 0.20. The achieved population ECE is reported.
* **Edges:** each premise edge is corrupted with probability r, by drop / rewire / add with equal chance.
* **Verifier mismatch:** the method assumes a perfect verifier when the true error is > 0, and 5% error when the true verifier is perfect.

## D23 — Stage C stopped before any API spending (2026-11-19, M5)
* **Decision:** Everything that runs without an LLM is built and tested: data download, gold graphs, corruption with truth by forward chaining, dependency F1, prompts and parsers, a call-counting client, and a mock. `GroqClient` refuses to send any request unless `BURHAVA_ALLOW_SPEND=1` is set. No paid call has been made.
* **Estimate** (`scripts/estimate_stage_c_cost.py`): 800 ProofWriter + 800 PrOntoQA instances (600 test + 200 calibration each) need about 62k calls, about 11.9M input tokens and about 4.0M output tokens. Tokens are approximated as characters / 4. Prices could not be fetched (the provider page is outside this environment's network allowlist).
* **Environment note:** the Groq credential injected for this session returned "Invalid API Key" on the free model-list endpoint.
* **Open design question for the researcher:** in ProofWriter/PrOntoQA the corrupted premises are fictional statements. An LLM's confidence in a premise therefore has almost no signal about whether it was corrupted. Confidence-based arms (3, 4) would then be near-random, which flatters arm 10. Options: (a) accept and say so; (b) corrupt premises with world-knowledge content, so confidence can carry signal; (c) give the reasoner noisy hints. This needs a decision before Stage C.
* **Stage B implication:** arm 10 loses its advantage once dependency-edge error reaches about 5%. So Stage C must report dependency-recovery F1 first. If F1 is low, H5 is expected to fail for reasons Stage B already explains.
