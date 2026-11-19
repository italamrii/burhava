"""Bayes risk rho, set risk R(S), and marginal value Delta, by exact enumeration.

Symbols
-------
* ``d``            final decision: ACCEPT (0), REJECT (1), ABSTAIN (2).
* ``L(d, y)``      loss of decision ``d`` when the conclusion's true value is ``y``.
* ``rho(E)``       Bayes risk given evidence E: the expected loss of the best
                   decision, ``min_d E[L(d, Y) | E]``.
* ``R(S)``         set risk: we will verify every atom in set ``S`` and then
                   decide. Before seeing the verdicts ``O_S`` we average over them:
                   ``R(S) = sum_{o_S} P(O_S = o_S | E) * rho(E, O_S = o_S)``.
* ``Delta_i(S)``   marginal value of adding atom ``i``:
                   ``Delta_i(S) = R(S) - R(S + {i})`` (always >= 0, see MATH.md).

How R(S) is computed fast
-------------------------
Write ``k = |S|`` and ``W = sum_z w(z)``. For a verdict pattern ``o``:

    P(o | E) * rho(E, o) = min_d  sum_z w(z) P(o | z) L(d, y(z)) / W
                         = min_d [ L(d,0) M0(o) + L(d,1) M1(o) ] / W

with ``M_y(o) = sum_{z: f(z)=y} w(z) prod_{i in S} P(o_i | z_i)``. The product
only depends on the k bits ``z_S``, so we first sum ``w`` into the ``2^k`` bins
of ``z_S`` (one pass over 2^n states), then push each bin through the 2x2
verifier channels (``2^k`` work per atom). Total cost ``O(2^n + k 2^k)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .model import Belief, channel

ACCEPT, REJECT, ABSTAIN = 0, 1, 2
DECISIONS = ("accept", "reject", "abstain")


class Loss:
    """Loss matrix ``L[d, y]`` for d in (accept, reject, abstain), y in (0, 1).

    Args:
        matrix: array-like of shape (3, 2).
    """

    def __init__(self, matrix: Sequence[Sequence[float]]):
        m = np.asarray(matrix, dtype=float)
        if m.shape != (3, 2):
            raise ValueError("loss matrix must have shape (3, 2)")
        if not np.all(np.isfinite(m)):
            raise ValueError("loss entries must be finite")
        self.matrix = m

    @classmethod
    def from_costs(cls, false_accept: float, false_reject: float, abstain: float) -> Loss:
        """Standard loss: correct decisions cost 0.

        ``L(accept, 0) = false_accept``, ``L(reject, 1) = false_reject``,
        ``L(abstain, y) = abstain`` for both y.
        """
        return cls([[false_accept, 0.0], [0.0, false_reject], [abstain, abstain]])

    @classmethod
    def zero_one(cls, abstain: float = 1.0) -> Loss:
        """0-1 loss. ``abstain=1`` (default) makes abstaining never strictly
        better than guessing, i.e. a plain binary accept/reject problem."""
        return cls.from_costs(1.0, 1.0, abstain)

    def __repr__(self) -> str:
        return f"Loss({self.matrix.tolist()})"


def _masses(belief: Belief) -> tuple[float, float]:
    """``(M0, M1)``: unnormalised posterior mass on Y=0 and on Y=1."""
    m1 = float(belief.w[belief.problem.y].sum())
    return belief.total - m1, m1


def expected_losses(belief: Belief, loss: Loss) -> np.ndarray:
    """``E[L(d, Y) | E]`` for each decision d, shape (3,)."""
    m0, m1 = _masses(belief)
    return (loss.matrix[:, 0] * m0 + loss.matrix[:, 1] * m1) / (m0 + m1)


def bayes_risk(belief: Belief, loss: Loss) -> float:
    """Bayes risk ``rho(E) = min_d E[L(d, Y) | E]``."""
    return float(expected_losses(belief, loss).min())


def bayes_decision(belief: Belief, loss: Loss) -> int:
    """The decision achieving ``rho(E)``. Ties go to the lower index
    (accept < reject < abstain)."""
    return int(np.argmin(expected_losses(belief, loss)))


def _check_set(belief: Belief, S: Sequence[int]) -> list[int]:
    S = [int(i) for i in S]
    if len(set(S)) != len(S):
        raise ValueError("S contains duplicates")
    n = belief.problem.n
    for i in S:
        if not 0 <= i < n:
            raise ValueError(f"atom index {i} out of range")
        if i in belief.evidence:
            raise ValueError(f"atom {i} is already verified")
    return S


def outcome_masses(belief: Belief, S: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``M0(o), M1(o)`` for every verdict pattern ``o`` of the atoms in ``S``.

    Returns two arrays of shape ``(2,) * k``; axis ``j`` is the verdict of
    atom ``S[j]``. ``M0 + M1`` summed over all patterns equals ``W``.
    """
    S = _check_set(belief, S)
    prob = belief.problem
    k = len(S)
    wy = belief.w * prob.y
    if k == 0:
        return np.array(belief.total - wy.sum()), np.array(wy.sum())
    # Bin index: atom S[j] contributes bit (k-1-j), so that after reshape to
    # (2,)*k, axis j holds z_{S[j]} (C-order puts the most significant bit first).
    code = np.zeros(prob.y.shape[0], dtype=np.int64)
    for j, i in enumerate(S):
        code |= prob.bits[i].astype(np.int64) << (k - 1 - j)
    m1 = np.bincount(code, weights=wy, minlength=1 << k).reshape((2,) * k)
    m0 = np.bincount(code, weights=belief.w - wy, minlength=1 << k).reshape((2,) * k)
    # Replace z_{S[j]} by verdict o_j along axis j: M[o] = sum_z C[o, z] M[z].
    for j, i in enumerate(S):
        C = channel(prob.alpha[i], prob.beta[i])
        m1 = np.moveaxis(np.tensordot(C, m1, axes=([1], [j])), 0, j)
        m0 = np.moveaxis(np.tensordot(C, m0, axes=([1], [j])), 0, j)
    return m0, m1


def set_risk(belief: Belief, loss: Loss, S: Iterable[int] = ()) -> float:
    """Set risk ``R(S) = E_{O_S}[ rho(E, O_S) ]`` (exact).

    Args:
        belief: current evidence ``E``.
        loss: loss matrix ``L``.
        S: atoms to be verified (not yet verified, no duplicates).

    Returns:
        Expected Bayes risk after observing the verdicts of ``S``.
    """
    m0, m1 = outcome_masses(belief, list(S))
    L = loss.matrix
    # per pattern o and decision d: L(d,0) M0(o) + L(d,1) M1(o); min over d.
    per_d = L[:, 0, None] * m0.reshape(1, -1) + L[:, 1, None] * m1.reshape(1, -1)
    return float(per_d.min(axis=0).sum() / belief.total)


def marginal_value(belief: Belief, loss: Loss, S: Iterable[int], i: int) -> float:
    """``Delta_i(S) = R(S) - R(S + {i})``: risk reduction from also verifying i."""
    S = list(S)
    return set_risk(belief, loss, S) - set_risk(belief, loss, S + [i])


# --------------------------------------------------------------------------
# Batched exact risks used by the policies (same maths, vectorised over atoms)
# --------------------------------------------------------------------------
def _channels(belief: Belief) -> np.ndarray:
    """``C[i, o, z] = P(O_i = o | Z_i = z)`` for all atoms, shape (n, 2, 2)."""
    a, b = belief.problem.alpha, belief.problem.beta
    C = np.empty((len(a), 2, 2))
    C[:, 0, 0], C[:, 0, 1] = 1 - a, b
    C[:, 1, 0], C[:, 1, 1] = a, 1 - b
    return C


def _min_over_decisions(loss: Loss, m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """``min_d L(d,0) m0 + L(d,1) m1`` elementwise."""
    L = loss.matrix
    out = L[0, 0] * m0 + L[0, 1] * m1
    for d in (1, 2):
        out = np.minimum(out, L[d, 0] * m0 + L[d, 1] * m1)
    return out


def single_risks(belief: Belief, loss: Loss) -> np.ndarray:
    """``R({i})`` for every atom i at once, shape (n,).

    Equals ``set_risk(belief, loss, [i])`` for unverified atoms (entries for
    already-verified atoms are meaningless and should be ignored).
    """
    p = belief.problem
    wy = belief.w * p.y
    w0 = belief.w - wy
    B = p.bits_f
    A1, A0 = B @ wy, B @ w0  # mass with z_i = 1, split by Y
    T1, T0 = wy.sum(), w0.sum()
    bin1 = np.stack([T1 - A1, A1], axis=1)  # (n, z)
    bin0 = np.stack([T0 - A0, A0], axis=1)
    C = _channels(belief)
    m1 = np.einsum("ioz,iz->io", C, bin1)
    m0 = np.einsum("ioz,iz->io", C, bin0)
    return _min_over_decisions(loss, m0, m1).sum(axis=1) / belief.total


def pair_risks(belief: Belief, loss: Loss) -> np.ndarray:
    """``R({i, j})`` for every pair, shape (n, n). Diagonal is meaningless.

    Masses for the four (z_i, z_j) bins come from inclusion-exclusion on
    ``P_y[i, j] = sum_z w(z) [f(z)=y] z_i z_j``.
    """
    p = belief.problem
    wy = belief.w * p.y
    w0 = belief.w - wy
    B = p.bits_f
    C = _channels(belief)
    out = np.zeros((p.n, p.n))
    ms = []
    for wv in (w0, wy):
        A = B @ wv
        P = (B * wv) @ B.T
        T = wv.sum()
        bins = np.empty((p.n, p.n, 2, 2))
        bins[:, :, 1, 1] = P
        bins[:, :, 1, 0] = A[:, None] - P
        bins[:, :, 0, 1] = A[None, :] - P
        bins[:, :, 0, 0] = T - A[:, None] - A[None, :] + P
        ms.append(np.einsum("iaz,jbx,ijzx->ijab", C, C, bins))
    out = _min_over_decisions(loss, ms[0], ms[1]).sum(axis=(2, 3)) / belief.total
    return out
