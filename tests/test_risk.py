import itertools

import numpy as np
import pytest
from random_graphs import random_graph, random_loss

from burhava import AssumptionGraph, Loss, Problem, bayes_decision, bayes_risk, set_risk
from burhava.risk import ABSTAIN, ACCEPT, REJECT


def naive_set_risk(b, L, S):
    """Definition-level R(S): enumerate verdict patterns, update, weigh."""
    total = 0.0
    for o in itertools.product([0, 1], repeat=len(S)):
        w = b.w.copy()
        for i, oi in zip(S, o, strict=True):
            w = w * b.problem.likelihood(i, oi)
        if w.sum() == 0:
            continue
        p_o = w.sum() / b.total
        m1 = w[b.problem.y].sum()
        exp = (L.matrix[:, 0] * (w.sum() - m1) + L.matrix[:, 1] * m1) / w.sum()
        total += p_o * exp.min()
    return total


def test_fast_matches_naive():
    rng = np.random.default_rng(1)
    for _ in range(200):
        prob = Problem(random_graph(rng))
        L = random_loss(rng)
        b = prob.prior_belief()
        S = [int(i) for i in rng.permutation(prob.n)[: int(rng.integers(0, prob.n + 1))]]
        assert set_risk(b, L, S) == pytest.approx(naive_set_risk(b, L, S), abs=1e-12)


def test_empty_set_equals_bayes_risk():
    rng = np.random.default_rng(2)
    for _ in range(50):
        b = Problem(random_graph(rng)).prior_belief()
        L = random_loss(rng)
        assert set_risk(b, L, []) == pytest.approx(bayes_risk(b, L))


def test_perfect_verifier_on_everything_gives_zero_risk():
    g = AssumptionGraph()
    for i in range(5):
        g.add_atom(f"z{i}", 0.3 + 0.1 * i)
    g.add_and("x", ["z0", "z1"])
    g.add_or("y", ["x", "z2", "z3"])
    g.add_and("c", ["y", "z4"])
    g.set_conclusion("c")
    b = Problem(g).prior_belief()
    assert set_risk(b, Loss.from_costs(10, 1, 0.6), range(5)) == pytest.approx(0.0)


def test_uninformative_verifier_has_zero_value():
    g = AssumptionGraph()
    g.add_atom("a", 0.7, alpha=0.4, beta=0.6)  # P(O=1|Z) = 0.4 regardless of Z
    g.add_atom("b", 0.5)
    g.add_and("y", ["a", "b"])
    g.set_conclusion("y")
    b = Problem(g).prior_belief()
    L = Loss.from_costs(10, 1, 0.6)
    assert set_risk(b, L, [0]) == pytest.approx(set_risk(b, L, []))


def test_decisions():
    g = AssumptionGraph()
    g.add_atom("a", 0.9)
    g.set_conclusion("a")
    b = Problem(g).prior_belief()
    assert bayes_decision(b, Loss.from_costs(10, 1, 0.6)) == ABSTAIN  # 1.0 vs .9 vs .6
    assert bayes_decision(b, Loss.from_costs(1, 1, 0.6)) == ACCEPT
    assert bayes_decision(b.observe(0, 0), Loss.from_costs(10, 1, 0.6)) == REJECT


def test_bad_sets_rejected():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    g.add_atom("b", 0.5)
    g.add_and("y", ["a", "b"])
    g.set_conclusion("y")
    b = Problem(g).prior_belief()
    L = Loss.zero_one()
    with pytest.raises(ValueError):
        set_risk(b, L, [0, 0])
    with pytest.raises(ValueError):
        set_risk(b.observe(0, 1), L, [0])
    with pytest.raises(ValueError):
        set_risk(b, L, [5])


def test_batched_risks_match_set_risk():
    from burhava.risk import pair_risks, single_risks

    rng = np.random.default_rng(3)
    for _ in range(60):
        prob = Problem(random_graph(rng))
        L = random_loss(rng)
        b = prob.prior_belief()
        if prob.n > 2 and rng.random() < 0.5:
            b = b.observe(0, int(rng.random() < b.prob_verdict(0)))
        s = single_risks(b, L)
        pr = pair_risks(b, L)
        free = [i for i in range(prob.n) if i not in b.evidence]
        for i in free:
            assert s[i] == pytest.approx(set_risk(b, L, [i]), abs=1e-12)
            for j in free:
                if i != j:
                    assert pr[i, j] == pytest.approx(set_risk(b, L, [i, j]), abs=1e-12)


def test_observe_y():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    g.set_conclusion("a")
    b = Problem(g).prior_belief().observe_y(1, 0.2)
    assert b.prob_y() == pytest.approx(0.8)
    assert b.y_obs == (1,)
