import numpy as np
import pytest
from random_graphs import random_graph, random_loss

from burhava import AssumptionGraph, Loss, Problem, set_risk
from burhava.generators import GENERATORS, GenConfig
from burhava.model import MixtureProblem
from burhava.noise import graph_ensemble
from burhava.policies import LookaheadVoI, PolicyState, VoIThenSpend, topo_order
from burhava.risk import pair_risks, single_risks


def _two_rule_graphs():
    gs = []
    for rule in ("and", "or"):
        g = AssumptionGraph()
        g.add_atom("a", 0.7)
        g.add_atom("b", 0.6)
        (g.add_and if rule == "and" else g.add_or)("y", ["a", "b"])
        g.set_conclusion("y")
        gs.append(g)
    return gs


def test_mixture_is_average_of_components():
    g_and, g_or = _two_rule_graphs()
    m = MixtureProblem([g_and, g_or], [0.25, 0.75])
    b = m.prior_belief()
    assert b.prob_y() == pytest.approx(0.25 * 0.42 + 0.75 * (1 - 0.3 * 0.4))
    assert b.prob_atom(0) == pytest.approx(0.7)
    # a single-graph mixture equals the plain problem
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = random_graph(rng)
        L = random_loss(rng)
        a, mm = Problem(g).prior_belief(), MixtureProblem([g]).prior_belief()
        S = list(range(min(2, g.n)))
        assert set_risk(mm, L, S) == pytest.approx(set_risk(a, L, S))


def test_batched_risks_on_mixture():
    g = GENERATORS["rules"](GenConfig("mixed", 4, 0.2), np.random.default_rng(1))
    m = MixtureProblem(graph_ensemble(g, 0.2, 4, np.random.default_rng(2)))
    b = m.prior_belief()
    L = Loss.from_costs(10, 1, 0.6)
    s, pr = single_risks(b, L), pair_risks(b, L)
    for i in range(min(m.n, 4)):
        assert s[i] == pytest.approx(set_risk(b, L, [i]))
        for j in range(i + 1, min(m.n, 4)):
            assert pr[i, j] == pytest.approx(set_risk(b, L, [i, j]))


def test_arm11_spends_budget_where_arm10_stops():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    g.add_atom("b", 0.5)  # irrelevant to Y
    g.add_or("y", ["a"])
    g.set_conclusion("y")
    p = Problem(g)
    b = p.prior_belief().observe(0, 1)  # Y is now known: no check has value

    def mk():
        return PolicyState(
            b, Loss.from_costs(10, 1, 0.6), np.random.default_rng(0), topo_order=topo_order(g)
        )

    assert LookaheadVoI().select(mk(), 3) is None
    assert VoIThenSpend().select(mk(), 3) == 1
