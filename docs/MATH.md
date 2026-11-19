# Mathematics of Burhava (plain language)

Every symbol is defined before it is used. Code locations are given for each formula.

## 1. Notation

| Symbol | Meaning | Code |
|---|---|---|
| `Z_i ∈ {0,1}` | atom *i*: an assumption, or "inference step is valid". 1 = true. | `graph.Atom` |
| `n` | number of atoms (at most 20) | `AssumptionGraph.n` |
| `z` | one *joint state*: a value for every atom. There are `N = 2^n` of them. State number `s` has bit *i* equal to `z_i`. | `Problem.bits` |
| `Y = f(z)` | the conclusion, computed from `z` by the AND/OR rules | `Problem.y` |
| `U_k` | hidden reliability of source *k*; takes values `u = 0, 1, …` with probability `P(U_k=u)` | `graph.Source` |
| `s(i)` | the source atom *i* cites (or none) | `Atom.source` |
| `P(z)` | prior probability of joint state `z` | `Problem.joint` |
| `c_i` | cost of verifying atom *i* | `Atom.cost` |
| `O_i` | the verifier's verdict on atom *i* (1 = "looks true") | |
| `α_i` | false-positive rate `P(O_i=1 ∣ Z_i=0)` | `Atom.alpha` |
| `β_i` | false-negative rate `P(O_i=0 ∣ Z_i=1)` | `Atom.beta` |
| `E` | evidence: the verdicts observed so far | `Belief.evidence` |
| `d` | final decision: accept, reject or abstain | `risk.ACCEPT/REJECT/ABSTAIN` |
| `L(d, y)` | loss of decision `d` when the truth is `y` | `risk.Loss` |
| `S` | a set of atoms we plan to verify | |

### Rules (the AND/OR graph)
* A **derivation** holds when *all* its premises hold **and** its step atom (if any) is 1. That is an AND.
* A **derived node** is true when *any* of its derivations holds. That is an OR.
* A premise may be negated ("NOT Z2"). We need this to write XOR, and ProofWriter uses negation too.

## 2. The prior P(z), including correlation

**Independent atoms.** Atom *i* is true with probability `p_i`, so its factor is `p_i` if `z_i=1` and `1−p_i` if `z_i=0`.

**Shared sources.** Atoms that cite the same source *k* share a hidden reliability `U_k`. Given `U_k = u`, each such atom is true with probability `P(Z_i=1 ∣ u)`, independently of the others. Averaging over the hidden `u` gives

```
P(z) = Π_k [ Σ_u P(U_k=u) · Π_{i cites k} P(z_i ∣ u) ]  ·  Π_{i cites no source} P(z_i)
```

In plain words, when a source is unreliable, all of its atoms tend to be false together. This product over sources is the same as the brief's formula `Σ_u P(u) Π_i P(Z_i ∣ U_{s(i)})`, because the sources are independent of each other. `tests/test_model.py::test_joint_matches_brute_force` checks the two forms agree. Code: `model.joint_table`.

## 3. Noisy verifier and Bayes update

The verifier is a 2×2 table `C[o, z] = P(O=o ∣ Z=z)`:

```
            Z=0      Z=1
  O=0     1−α        β
  O=1       α      1−β
```

Bayes' rule says that after seeing `O_i = o`, every joint state's weight is multiplied by how likely that verdict was in that state:

```
w_new(z) = w(z) · P(O_i = o ∣ z_i)
```

We keep `w` *unnormalised*. Its total `W = Σ_z w(z)` is the probability of all evidence so far. A probability is always a ratio, for example `P(Y=1 ∣ E) = Σ_{z: f(z)=1} w(z) / W`. Code: `Belief.observe`.

## 4. Risk

**Bayes risk** is the expected loss if we decide now, choosing the best decision:

```
ρ(E) = min_d  E[ L(d, Y) ∣ E ] = min_d [ L(d,0)·P(Y=0∣E) + L(d,1)·P(Y=1∣E) ]
```

**Set risk.** We plan to verify every atom in `S`, see the verdicts `O_S`, and then decide. Before we see the verdicts, we average over what they might be:

```
R(S) = Σ_{o_S} P(O_S = o_S ∣ E) · ρ(E, O_S = o_S)
```

**Marginal value** of atom *i* given that `S` will be checked anyway:

```
Δ_i(S) = R(S) − R(S ∪ {i})
```

This is classical *Value of Information* (Howard 1966; see RELATED_WORK.md). Burhava does not propose a new principle. It tests empirically whether choosing checks by `Δ_i/c_i` beats choosing them by confidence.

**Fast exact computation** (`risk.outcome_masses`). Write `M_y(o) = Σ_{z: f(z)=y} w(z) Π_{i∈S} P(o_i ∣ z_i)`. Then

```
R(S) = (1/W) · Σ_o  min_d [ L(d,0)·M_0(o) + L(d,1)·M_1(o) ]
```

The product depends only on the |S| bits `z_S`. So we (1) add `w` into `2^|S|` bins by `z_S`, which is one pass over all `2^n` states, and (2) push each bin through the 2×2 verifier tables, one atom at a time. The cost is `O(2^n + |S|·2^|S|)`. The test `test_risk.py::test_fast_matches_naive` compares this against the literal definition (enumerate each `o_S`, Bayes-update, weigh) on 200 random graphs.

## 5. The four acceptance facts, step by step

### 5.1 AND counterexample
Setup: `Y = Z1 AND Z2`, `P(Z1=1) = P(Z2=1) = 0.5`, independent. The verifier is perfect (`α=β=0`). The loss is 0–1: a wrong accept or a wrong reject costs 1. Abstaining costs 1, so it never helps.

* **R(∅).** `P(Y=1) = 0.5·0.5 = 0.25`. Rejecting is wrong with probability 0.25 and accepting with probability 0.75, so `R(∅) = 0.25`.
* **R({1}).** With probability 0.5 we see `Z1=0`. Then `Y=0` for sure, so we reject and the loss is 0. With probability 0.5 we see `Z1=1`. Then `Y = Z2`, so `P(Y=1)=0.5` and any guess is wrong half the time. In total `R({1}) = 0.5·0 + 0.5·0.5 = 0.25`. By symmetry `R({2}) = 0.25`.
* **R({1,2}).** Y is now known exactly, so `R = 0`.

**Why it matters.** Each single check has value `Δ_1(∅) = Δ_2(∅) = 0`, but the pair is worth 0.25. A greedy one-step rule sees zero value everywhere and stops (it "stalls"). A two-step lookahead sees the pair's value. This is the M2 test that arm 9 stalls and arm 10 does not. In technical terms, `Δ` is not submodular here.

### 5.2 XOR
Setup: `Y = Z1 XOR Z2`, written as `(Z1 AND NOT Z2) OR (NOT Z1 AND Z2)`, with everything else as in 5.1.
* `P(Y=1) = 0.5`, so `R(∅) = 0.5`.
* Seeing `Z1` alone leaves `Y = NOT Z2` or `Y = Z2`, and both are still 50/50. So `R({1}) = R({2}) = 0.5`.
* Seeing both gives `R({1,2}) = 0`.

### 5.3 Selection example
Setup: `Y = Z1 AND (Z2 OR Z3)`, priors `(0.90, 0.60, 0.55)`, independent, perfect verifier. Losses: a false accept costs 10, a false reject costs 1, abstaining costs 0.6.

Expected loss of each decision when `P(Y=1) = q`: accept `10(1−q)`, reject `q`, abstain `0.6`.

* **R(∅).** `P(Z2 OR Z3) = 1 − 0.4·0.45 = 0.82`, so `q = 0.9·0.82 = 0.738`. The options are accept 2.62, reject 0.738 and abstain 0.6, so `R(∅) = 0.6`.
* **Check Z1.** If `Z1=1` (prob 0.9): `q = 0.82`, options 1.8 / 0.82 / 0.6, best 0.6. If `Z1=0` (prob 0.1): `Y=0`, reject, loss 0. So `R({1}) = 0.54` and **`Δ_1 = 0.060`**.
* **Check Z2.** If `Z2=1` (0.6): `q = 0.9`, options 1.0 / 0.9 / 0.6, best 0.6. If `Z2=0` (0.4): `q = 0.9·0.55 = 0.495`, best is reject at 0.495. So `R({2}) = 0.36 + 0.198 = 0.558` and **`Δ_2 = 0.042`**.
* **Check Z3.** If `Z3=1` (0.55): `q = 0.9`, best 0.6. If `Z3=0` (0.45): `q = 0.54`, best is reject at 0.54. So `R({3}) = 0.33 + 0.243 = 0.573` and **`Δ_3 = 0.027`**.

Z1 has the *highest* confidence (0.90) yet the *largest* value of checking. This is because a false Z1 kills the conclusion outright, and a false accept is expensive. A lowest-confidence-first rule would check Z3 first, the least valuable atom. The code reproduces these numbers exactly: `R = 0.6, (0.54, 0.558, 0.573)`.

### 5.4 Monotonicity: `R(S ∪ {i}) ≤ R(S)`
**Claim.** Under the model's own beliefs, extra information never increases expected risk. This holds for any graph, loss matrix or verifier error rates.

**Why.**
1. For any fixed decision `d`, its expected loss is a *weighted average* of the losses (weights = posterior probabilities). The Bayes risk is the *minimum* of these averages.
2. The minimum of averages is "concave": averaging first and then taking the minimum is ≥ taking the minimum in each case and then averaging. (You can only do better if you are allowed to choose after seeing more.)
3. Before we see `O_i`, the posterior we expect to hold *after* seeing it averages back to today's posterior. This is the law of total probability.
4. Combining 2 and 3 gives `E[ρ after O_i] ≤ ρ now`. Averaging this over `O_S` gives the claim.

**Caveat (matters for M4).** This is a statement about the *model's* risk. If the model's probabilities are wrong (miscalibrated, wrong edges, wrong α/β), the *realised* loss on the true world can go up after a check. Stage B measures this. The property test (`test_acceptance.py::test_monotonicity`) runs 500 random cases: random graphs with sources, steps and negation; random loss entries in [0, 10]; `α, β` anywhere in [0, 1]; and random prior evidence.

## 6. Complexity and timing
At `n = 20` atoms (2^20 ≈ 1M states), building the joint table takes 0.14 s. Each `R(S)` then takes 4–21 ms for |S| = 0…5 (EXPERIMENTS.md, entry M1-timing).

## 7. The ten selection arms (M2)
`p_i = P(Z_i=1 ∣ E)` is the current confidence in atom *i*. `Δ_i = ρ(E) − R({i})` is the risk removed by checking *i* alone.

| Arm | Picks next | Uses loss? | Uses cost? | Stops when |
|---|---|---|---|---|
| 1 none | nothing | – | – | immediately |
| 2 random | a random affordable atom | no | no | budget spent |
| 3 lowest-confidence | lowest `p_i` | no | no | budget spent |
| 4 highest-entropy | `p_i` closest to 0.5 | no | no | budget spent |
| 5 highest-degree | the atom most derivations use | no | no | budget spent |
| 6 self-consistency | a noisy sample of Y (error 0.3) | yes (final decision) | – | budget spent |
| 7 topological | first unchecked atom in chain order | no | no | first negative verdict |
| 8 propagation | largest expected shift of P(Y=1) | no | no | budget spent |
| 9 one-step VoI | largest `Δ_i / c_i` | **yes** | **yes** | no check has value |
| 10 lookahead VoI | as 9, but if the best `Δ_i/c_i ≤ 0.05·ρ`, also scores pairs `(ρ − R({i,j}))/(c_i+c_j)` | **yes** | **yes** | no single or pair has value |

All arms make the same final decision rule, the Bayes decision under their beliefs.

**Why arm 10 exists.** On the AND counterexample (§5.1), every single check has `Δ_i = 0`, so arm 9 stops at once with loss 0.25. Arm 10 sees that the pair is worth 0.25, checks Z1, and then (if Z1 = 1) checks Z2. With a perfect verifier it is always right (`tests/test_policies.py`).
