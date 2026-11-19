"""Corrupting the model the method sees (Stage A: correlation; Stage B: noise).

Ground truth is never changed; every function here returns a *new* graph with
the same atoms in the same order, which becomes the method's ``model``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .graph import AssumptionGraph, rebuild


def independent_version(graph: AssumptionGraph) -> AssumptionGraph:
    """Drop all sources: each atom keeps its true *marginal* prior
    ``P(Z_i=1) = sum_u P(U=u) P(Z_i=1|u)`` but correlations are ignored."""
    atoms = []
    for a in graph.atoms:
        if a.source is None:
            atoms.append(a)
        else:
            pu = graph.sources[a.source].u_probs
            marg = sum(p * q for p, q in zip(pu, a.prior, strict=True))
            atoms.append(replace(a, source=None, prior=(float(marg),)))
    return rebuild(graph, atoms=atoms, sources={})


# ---------------------------------------------------------------- Stage B
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def perturb_calibration(graph: AssumptionGraph, sigma: float, eps: np.ndarray) -> AssumptionGraph:
    """Miscalibrate every atom prior in logit space.

    ``logit(p'_i) = logit(p_i) + sigma * eps_i`` with ``eps_i ~ N(0, 1)`` given
    by the caller (so the same draw can be reused across noise levels: paired).
    For a source-correlated atom, the same shift is applied to each
    ``P(Z_i = 1 | u)``. Priors are clipped to [0.001, 0.999].
    """
    atoms = []
    for i, a in enumerate(graph.atoms):
        q = np.clip(_sigmoid(_logit(a.prior) + sigma * eps[i]), 0.001, 0.999)
        atoms.append(replace(a, prior=tuple(float(x) for x in q)))
    return rebuild(graph, atoms=atoms)


def marginal_priors(graph: AssumptionGraph) -> np.ndarray:
    """``P(Z_i = 1)`` for every atom (averaging over its source if any)."""
    out = []
    for a in graph.atoms:
        if a.source is None:
            out.append(a.prior[0])
        else:
            pu = graph.sources[a.source].u_probs
            out.append(sum(p * q for p, q in zip(pu, a.prior, strict=True)))
    return np.array(out)


def population_ece(p_true: np.ndarray, p_model: np.ndarray, n_bins: int = 10) -> float:
    """ECE of model probabilities against the *true* probabilities.

    Same binning as :func:`burhava.stats.ece`, but uses ``E[Z] = p_true``
    instead of sampled 0/1 outcomes, so there is no sampling noise.
    """
    b = np.minimum((p_model * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p_true[m].mean() - p_model[m].mean())
    return float(tot / p_model.size)


def sigma_for_ece(p_true: np.ndarray, eps: np.ndarray, target: float) -> float:
    """Bisection for the logit-noise scale whose population ECE hits ``target``."""
    if target <= 0:
        return 0.0
    lt = _logit(p_true)

    def f(s):
        return population_ece(p_true, np.clip(_sigmoid(lt + s * eps), 0.001, 0.999))

    lo, hi = 0.0, 1.0
    while f(hi) < target:
        hi *= 2
        if hi > 64:
            raise ValueError(f"ECE {target} not reachable")
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def corrupt_edges(
    graph: AssumptionGraph, rate: float, rng: np.random.Generator
) -> tuple[AssumptionGraph, int, int]:
    """Simulate dependency-recovery errors.

    Each premise edge is corrupted with probability ``rate`` by one of three
    equally likely operations:
    * **drop**: remove the premise (only if the derivation keeps a premise or step;
      otherwise it is rewired instead);
    * **rewire**: point it at another node defined earlier (an atom or a derived node);
    * **add**: keep it, and also add a spurious premise on an earlier node.

    Returns ``(new_graph, n_edges, n_changed)``. Atoms are unchanged, so indices stay aligned.
    """
    from .graph import Derivation, DerivedNode, Literal

    if rate <= 0:
        n_e = sum(len(d.premises) for nd in graph.derived for d in nd.derivations)
        return graph, n_e, 0
    defined = [a.name for a in graph.atoms]
    new_nodes, n_edges, n_changed = [], 0, 0
    for nd in graph.derived:
        ders = []
        for d in nd.derivations:
            prem = list(d.premises)
            out: list[Literal] = []
            for k, lit in enumerate(prem):
                n_edges += 1
                if rng.random() >= rate:
                    out.append(lit)
                    continue
                n_changed += 1
                op = rng.choice(["drop", "rewire", "add"])
                others = [x for x in defined if x != lit.node]
                remaining = len(prem) - k - 1 + len(out)
                if op == "drop" and (remaining > 0 or d.step is not None):
                    continue
                if op == "add":
                    out.append(lit)
                    out.append(Literal(str(rng.choice(others))))
                else:  # rewire (or a drop that would leave the derivation empty)
                    out.append(Literal(str(rng.choice(others)), lit.negated))
            # remove duplicate premises, keep order
            seen, uniq = set(), []
            for lit in out:
                if (lit.node, lit.negated) not in seen:
                    seen.add((lit.node, lit.negated))
                    uniq.append(lit)
            ders.append(Derivation(tuple(uniq), d.step))
        new_nodes.append(DerivedNode(nd.name, tuple(ders)))
        defined.append(nd.name)
    return rebuild(graph, derived=new_nodes), n_edges, n_changed


def with_verifier(graph: AssumptionGraph, error: float) -> AssumptionGraph:
    """Copy of ``graph`` with ``alpha_i = beta_i = error`` for every atom."""
    return rebuild(graph, atoms=[replace(a, alpha=error, beta=error) for a in graph.atoms])


def graph_ensemble(
    recovered: AssumptionGraph, r_hat: float, k: int, rng: np.random.Generator
) -> list[AssumptionGraph]:
    """The recovered graph plus ``k - 1`` copies with edges re-corrupted at
    rate ``r_hat``: a cheap sample of 'graphs that might be the true one'."""
    return [recovered] + [corrupt_edges(recovered, r_hat, rng)[0] for _ in range(k - 1)]
