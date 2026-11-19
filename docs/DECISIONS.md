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
