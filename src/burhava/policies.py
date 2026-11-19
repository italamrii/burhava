"""The 10 verification-selection arms.

Every arm has the same interface::

    policy.select(state, budget_left) -> atom index | SAMPLE_Y | None

* an atom index ``i``: verify atom ``i`` next;
* ``SAMPLE_Y`` (only arm 6): draw one more noisy sample of the conclusion;
* ``None``: stop verifying and decide.

An atom is a *candidate* if it is verifiable, not yet verified, and its cost
``c_i`` fits in ``budget_left``. Ties are always broken by lower cost, then lower
index (see ``_argbest``).

Symbols: ``p_i = P(Z_i = 1 | E)`` (current confidence), ``rho`` = current Bayes
risk, ``Delta_i = rho - R({i})`` (one-step value), ``c_i`` = cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Belief
from .risk import Loss, bayes_risk, pair_risks, single_risks

SAMPLE_Y = -1
_TOL = 1e-12


@dataclass
class PolicyState:
    """What a policy may look at when choosing the next check.

    Attributes:
        belief: current posterior (under the model the method believes).
        loss: loss matrix ``L``.
        rng: the policy's own random generator (only arm 2 uses it).
        verdicts: ``{i: o_i}`` observed so far (same as ``belief.evidence``).
        n_risk_evals: running count of R(S) values computed (cost metric).
    """

    belief: Belief
    loss: Loss
    rng: np.random.Generator
    n_risk_evals: int = 0
    topo_order: list[int] = field(default_factory=list)

    @property
    def verdicts(self) -> dict[int, int]:
        return self.belief.evidence


def candidates(state: PolicyState, budget_left: float) -> np.ndarray:
    """Indices of verifiable, unverified atoms with ``c_i <= budget_left``."""
    p = state.belief.problem
    ok = p.verifiable & (p.cost <= budget_left + 1e-9)
    for i in state.belief.evidence:
        ok[i] = False
    return np.flatnonzero(ok)


def _argbest(score: np.ndarray, cand: np.ndarray, cost: np.ndarray) -> int:
    """Candidate with the highest score; ties -> lower cost, then lower index."""
    s = score[cand]
    best = s.max()
    tied = cand[np.isclose(s, best, rtol=1e-12, atol=1e-15)]
    order = np.lexsort((tied, cost[tied]))  # primary key: cost, secondary: index
    return int(tied[order[0]])


def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


class Policy:
    """Base class. ``name`` is the arm label used in results files."""

    name = "base"

    def select(self, state: PolicyState, budget_left: float) -> int | None:
        raise NotImplementedError


class NoVerification(Policy):
    """Arm 1: never verify; decide from the prior."""

    name = "a01_none"

    def select(self, state, budget_left):
        return None


class RandomPolicy(Policy):
    """Arm 2: a uniformly random candidate."""

    name = "a02_random"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        return None if cand.size == 0 else int(state.rng.choice(cand))


class LowestConfidence(Policy):
    """Arm 3: the candidate with the lowest ``p_i = P(Z_i = 1 | E)``."""

    name = "a03_lowconf"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        p = state.belief.prob_atoms()
        return _argbest(-p, cand, state.belief.problem.cost)


class HighestEntropy(Policy):
    """Arm 4: the candidate with the highest binary entropy ``H(p_i)``."""

    name = "a04_entropy"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        return _argbest(_entropy(state.belief.prob_atoms()), cand, state.belief.problem.cost)


class HighestDegree(Policy):
    """Arm 5: the candidate used by the most derivations (graph out-degree)."""

    name = "a05_degree"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        deg = state.belief.problem.graph.degree().astype(float)
        return _argbest(deg, cand, state.belief.problem.cost)


class SelfConsistency(Policy):
    """Arm 6: ignore the atoms; spend budget on noisy samples of the conclusion.

    Each sample costs ``sample_cost`` and says ``Y`` correctly with probability
    ``1 - eps``; samples are independent given ``Y`` (a generous assumption for
    real self-consistency). The decision uses the exact Bayes posterior.
    """

    name = "a06_selfcons"

    def __init__(self, eps: float = 0.3, sample_cost: float = 1.0):
        self.eps = eps
        self.sample_cost = sample_cost

    def select(self, state, budget_left):
        return SAMPLE_Y if budget_left + 1e-9 >= self.sample_cost else None


class TopologicalStopAtFailure(Policy):
    """Arm 7: verify atoms in the order the chain first uses them; stop at the
    first negative verdict (the chain is considered broken)."""

    name = "a07_topo"

    def select(self, state, budget_left):
        if any(o == 0 for o in state.belief.evidence.values()):
            return None
        cand = set(candidates(state, budget_left).tolist())
        for i in state.topo_order:
            if i in cand:
                return i
        return None


class PropagationScore(Policy):
    """Arm 8: loss-unaware influence score.

    ``score_i = 2 p_i (1 - p_i) |P(Y=1 | Z_i=1, E) - P(Y=1 | Z_i=0, E)|`` =
    expected absolute change of ``P(Y=1)`` if ``Z_i`` were revealed. It uses the
    graph and the probabilities but ignores the loss and the costs.
    """

    name = "a08_propagation"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        b = state.belief
        B = b.problem.bits_f
        W = b.total
        a = B @ b.w  # mass with z_i = 1
        a1 = B @ (b.w * b.problem.y)  # ... and Y = 1
        t1 = float(b.w[b.problem.y].sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            py_given_1 = np.where(a > 0, a1 / a, 0.0)
            py_given_0 = np.where(W - a > 0, (t1 - a1) / (W - a), 0.0)
        p = a / W
        score = 2 * p * (1 - p) * np.abs(py_given_1 - py_given_0)
        return _argbest(score, cand, b.problem.cost)


class OneStepVoI(Policy):
    """Arm 9: greedy ``Delta_i / c_i``; stop when no check has positive value.

    This is myopic Value of Information. It *stalls* when every single check is
    worthless on its own even though a pair is valuable (AND counterexample).
    """

    name = "a09_voi1"

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        b, L = state.belief, state.loss
        rho = bayes_risk(b, L)
        r1 = single_risks(b, L)
        state.n_risk_evals += 1 + int(cand.size)
        score = (rho - r1) / b.problem.cost
        if score[cand].max() <= _TOL:
            return None
        return _argbest(score, cand, b.problem.cost)


class LookaheadVoI(Policy):
    """Arm 10: ``Delta_i / c_i`` with a two-step lookahead when it is small.

    1. Compute one-step scores ``s_i = Delta_i / c_i``.
    2. If ``max s_i <= eta * rho`` (the best single check removes at most a
       fraction ``eta`` of the current risk per unit cost) and the budget allows
       two checks, also score every affordable pair:
       ``s_ij = (rho - R({i, j})) / (c_i + c_j)``.
    3. If the best pair rate beats the best single rate, verify the member of
       that pair with the higher single score (ties: cost, index); the next call
       re-plans with the new verdict. Otherwise behave like arm 9.
    """

    name = "a10_voi2"

    def __init__(self, eta: float = 0.05):
        self.eta = eta

    def select(self, state, budget_left):
        cand = candidates(state, budget_left)
        if cand.size == 0:
            return None
        b, L = state.belief, state.loss
        cost = b.problem.cost
        rho = bayes_risk(b, L)
        r1 = single_risks(b, L)
        state.n_risk_evals += 1 + int(cand.size)
        s1 = (rho - r1) / cost
        best1 = s1[cand].max()
        if best1 <= self.eta * rho and cand.size >= 2:
            r2 = pair_risks(b, L)
            cc = cost[cand]
            ok = (cc[:, None] + cc[None, :] <= budget_left + 1e-9) & ~np.eye(cand.size, dtype=bool)
            n_pairs = int(np.triu(ok).sum())
            state.n_risk_evals += n_pairs
            if n_pairs:
                sub = r2[np.ix_(cand, cand)]
                s2 = np.where(ok, (rho - sub) / (cc[:, None] + cc[None, :]), -np.inf)
                best2 = s2.max()
                if best2 > max(best1, _TOL) + 1e-15:
                    # deterministic choice among best pairs: lowest (i, j) in index order
                    ii, jj = np.nonzero(np.isclose(s2, best2, rtol=1e-12, atol=1e-15))
                    k = int(np.lexsort((jj, ii))[0])
                    pair = np.array([cand[ii[k]], cand[jj[k]]])
                    return _argbest(s1, pair, cost)
        if best1 <= _TOL:
            return None
        return _argbest(s1, cand, cost)


class VoIThenSpend(LookaheadVoI):
    """Arm 11 (phase 2): arm 10 while it finds positive value; once it would
    stop with budget left, spend the remainder by propagation score (arm 8).

    Motivation (Stage A, exploratory): arm 10 stops early, and the loss-blind
    arm 8 beat it at budgets 3-5. Under a correct model, extra checks never
    raise the expected risk (MATH.md section 5.4), so spending the budget is free.
    """

    name = "a11_voi_spend"

    def select(self, state, budget_left):
        i = super().select(state, budget_left)
        return PropagationScore().select(state, budget_left) if i is None else i


def topo_order(graph) -> list[int]:
    """Atoms in the order the chain first uses them (premises, then step),
    scanning derived nodes in definition order; unused atoms go last."""
    seen: list[int] = []
    for node in graph.derived:
        for d in node.derivations:
            names = [lit.node for lit in d.premises] + ([d.step] if d.step else [])
            for nm in names:
                kind, idx = graph._index[nm]
                if kind == "atom" and idx not in seen:
                    seen.append(idx)
    return seen + [i for i in range(graph.n) if i not in seen]


def all_arms(eta: float = 0.05, sc_eps: float = 0.3) -> list[Policy]:
    """The 10 arms in order."""
    return [
        NoVerification(),
        RandomPolicy(),
        LowestConfidence(),
        HighestEntropy(),
        HighestDegree(),
        SelfConsistency(eps=sc_eps),
        TopologicalStopAtFailure(),
        PropagationScore(),
        OneStepVoI(),
        LookaheadVoI(eta=eta),
    ]


def arms_for(cfg_arms: dict) -> list[Policy]:
    """Arms named in ``cfg_arms["include"]`` (default: the 10 original arms)."""
    pool = all_arms(eta=cfg_arms["eta"], sc_eps=cfg_arms["sc_eps"])
    pool.append(VoIThenSpend(eta=cfg_arms["eta"]))
    names = cfg_arms.get("include")
    if not names:
        return pool[:10]
    by = {a.name: a for a in pool}
    return [by[n] for n in names]
