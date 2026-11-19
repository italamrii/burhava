"""Joint distribution P(Z), verifier noise, and Bayes updating.

Notation (all symbols are defined here and in docs/MATH.md)
-----------------------------------------------------------
* ``n``            number of atoms; ``N = 2^n`` joint states.
* ``z``            one joint state, a vector of n bits; state index ``s`` has
                   bit ``i`` equal to ``z_i``.
* ``P(z)``         prior probability of state ``z`` (the "joint table").
* ``U_k``          hidden reliability of source ``k``, with ``P(U_k = u)``.
* ``P(Z_i=1|u)``   atom ``i``'s probability of being true when its source has
                   reliability ``u``.
* Correlation model::

      P(z) = sum over u of P(u) * prod_i P(z_i | U_{s(i)} = u_{s(i)})

  Sources are independent of each other, so this factorises source by source:
  P(z) = prod_k [ sum_{u_k} P(u_k) prod_{i cites k} P(z_i | u_k) ]
         * prod_{i cites no source} P(z_i).
* Verifier: checking atom ``i`` returns verdict ``O_i`` with
  ``P(O_i=1 | Z_i=0) = alpha_i`` (false positive) and
  ``P(O_i=0 | Z_i=1) = beta_i`` (false negative). Verdicts are independent given Z.
* Evidence ``E``: the verdicts seen so far. The posterior is
  ``P(z | E) proportional to P(z) * prod_{(i,o) in E} P(O_i = o | z_i)``.

We store the *unnormalised* posterior ``w(z) = P(z) * likelihood`` and divide by
``W = sum_z w(z)`` only when a probability is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import AssumptionGraph


def channel(alpha: float, beta: float) -> np.ndarray:
    """Verifier channel matrix ``C[o, z] = P(O = o | Z = z)``.

    Returns:
        2x2 array ``[[1-alpha, beta], [alpha, 1-beta]]``.
    """
    return np.array([[1.0 - alpha, beta], [alpha, 1.0 - beta]])


class Problem:
    """A graph with its exact joint table, cached once per instance.

    Attributes:
        graph: the :class:`AssumptionGraph`.
        n: number of atoms.
        bits: bool array ``(n, 2^n)``; ``bits[i, s] = z_i`` in state ``s``.
        joint: float array ``(2^n,)``; ``joint[s] = P(z)``.
        y: bool array ``(2^n,)``; ``y[s] = f(z)``, the conclusion in state ``s``.
    """

    def __init__(self, graph: AssumptionGraph):
        if graph.conclusion is None:
            raise ValueError("graph has no conclusion")
        self.graph = graph
        self.n = graph.n
        n = self.n
        states = np.arange(1 << n, dtype=np.uint32)
        self.bits = ((states[None, :] >> np.arange(n, dtype=np.uint32)[:, None]) & 1).astype(bool)
        self.joint = joint_table(graph, self.bits)
        self.y = graph.conclusion_values(self.bits)
        self.cost = np.array([a.cost for a in graph.atoms])
        self.alpha = np.array([a.alpha for a in graph.atoms])
        self.beta = np.array([a.beta for a in graph.atoms])
        self.verifiable = np.array([a.verifiable for a in graph.atoms], dtype=bool)

    def likelihood(self, i: int, o: int) -> np.ndarray:
        """``P(O_i = o | z)`` for every state ``z`` (shape ``(2^n,)``)."""
        a, b = self.alpha[i], self.beta[i]
        if o == 1:
            return np.where(self.bits[i], 1.0 - b, a)
        return np.where(self.bits[i], b, 1.0 - a)

    def prior_belief(self) -> Belief:
        """The belief before any verdict: ``w = P(z)``, ``E`` empty."""
        return Belief(self, self.joint.copy(), {})


def joint_table(graph: AssumptionGraph, bits: np.ndarray) -> np.ndarray:
    """Exact prior ``P(z)`` for every state, using the shared-source model.

    Args:
        graph: the assumption graph (atoms, sources, priors).
        bits: bool array ``(n, 2^n)``.

    Returns:
        float array ``(2^n,)`` summing to 1.
    """
    N = bits.shape[1]
    p = np.ones(N)
    # Independent atoms: multiply P(z_i).
    for i, a in enumerate(graph.atoms):
        if a.source is None:
            q = a.prior[0]
            p *= np.where(bits[i], q, 1.0 - q)
    # Each source k: one mixture factor over its latent values u_k.
    for name, src in graph.sources.items():
        members = [i for i, a in enumerate(graph.atoms) if a.source == name]
        if not members:
            continue
        factor = np.zeros(N)
        for u, pu in enumerate(src.u_probs):
            term = np.full(N, pu)
            for i in members:
                q = graph.atoms[i].prior[u]
                term *= np.where(bits[i], q, 1.0 - q)
            factor += term
        p *= factor
    return p


@dataclass
class Belief:
    """Current knowledge: unnormalised posterior ``w(z)`` and evidence ``E``.

    Attributes:
        problem: the :class:`Problem` the method believes in (may differ from the
            true world in noise experiments).
        w: unnormalised posterior weights, shape ``(2^n,)``.
        evidence: ``{atom index i: observed verdict o_i}``.
    """

    problem: Problem
    w: np.ndarray
    evidence: dict[int, int] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """``W = sum_z w(z)`` = probability of the evidence seen so far."""
        return float(self.w.sum())

    def observe(self, i: int, o: int) -> Belief:
        """Bayes update after verdict ``O_i = o``: ``w'(z) = w(z) * P(O_i=o | z_i)``.

        Returns a new :class:`Belief`; ``self`` is unchanged.
        """
        if i in self.evidence:
            raise ValueError(f"atom {i} already verified")
        if o not in (0, 1):
            raise ValueError("verdict must be 0 or 1")
        w = self.w * self.problem.likelihood(i, o)
        if w.sum() <= 0:
            raise ValueError(f"verdict O_{i}={o} has probability 0 under the model")
        return Belief(self.problem, w, {**self.evidence, i: o})

    def prob_atom(self, i: int) -> float:
        """Posterior marginal ``P(Z_i = 1 | E)``."""
        return float(self.w[self.problem.bits[i]].sum() / self.total)

    def prob_atoms(self) -> np.ndarray:
        """``P(Z_i = 1 | E)`` for all atoms, shape ``(n,)``."""
        # A loop avoids materialising a float copy of the (n, 2^n) bit matrix.
        W = self.total
        return np.array([self.w[b].sum() / W for b in self.problem.bits])

    def prob_y(self) -> float:
        """Posterior probability of the conclusion, ``P(Y = 1 | E)``."""
        return float(self.w[self.problem.y].sum() / self.total)

    def prob_verdict(self, i: int) -> float:
        """Predictive probability of a positive verdict, ``P(O_i = 1 | E)``."""
        return float(self.w @ self.problem.likelihood(i, 1) / self.total)
