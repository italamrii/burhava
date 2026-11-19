import itertools

import numpy as np
from random_graphs import random_graph

from burhava import AssumptionGraph, Problem


def _brute_joint(g):
    """P(z) by explicitly enumerating every combination of latent values u."""
    names = list(g.sources)
    n = g.n
    out = np.zeros(1 << n)
    for us in itertools.product(*[range(len(g.sources[k].u_probs)) for k in names]):
        pu = np.prod([g.sources[k].u_probs[u] for k, u in zip(names, us, strict=True)])
        u_of = dict(zip(names, us, strict=True))
        for s in range(1 << n):
            p = pu
            for i, a in enumerate(g.atoms):
                q = a.prior[0] if a.source is None else a.prior[u_of[a.source]]
                p *= q if (s >> i) & 1 else 1 - q
            out[s] += p
    return out


def test_joint_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        g = random_graph(rng, max_atoms=6)
        np.testing.assert_allclose(Problem(g).joint, _brute_joint(g), atol=1e-14)
        assert Problem(g).joint.sum() == np.float64(1.0) or abs(Problem(g).joint.sum() - 1) < 1e-12


def test_shared_source_induces_correlation():
    g = AssumptionGraph()
    g.add_source("s", [0.5, 0.5])  # u=0 unreliable, u=1 reliable
    g.add_atom("a", (0.2, 0.9), source="s")
    g.add_atom("b", (0.2, 0.9), source="s")
    g.add_and("y", ["a", "b"])
    g.set_conclusion("y")
    b = Problem(g).prior_belief()
    pa = b.prob_atom(0)
    assert np.isclose(pa, 0.55)
    # Seeing a=1 makes b=1 more likely: P(b|a) = (.5*.04+.5*.81)/.55
    post = b.observe(0, 1)
    assert np.isclose(post.prob_atom(1), (0.5 * 0.04 + 0.5 * 0.81) / 0.55)
    assert post.prob_atom(1) > pa


def test_bayes_update_noisy_verifier():
    g = AssumptionGraph()
    g.add_atom("a", 0.3, alpha=0.1, beta=0.2)
    g.set_conclusion("a")
    b = Problem(g).prior_belief()
    assert np.isclose(b.prob_verdict(0), 0.3 * 0.8 + 0.7 * 0.1)
    post = b.observe(0, 1)
    assert np.isclose(post.prob_atom(0), 0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.1))
    assert np.isclose(post.prob_y(), post.prob_atom(0))
    np.testing.assert_allclose(b.prob_atoms(), [0.3])
