"""The four acceptance tests from the research brief (Section 1).

Each test reproduces a known fact exactly; docs/MATH.md derives every number by hand.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from random_graphs import random_graph, random_loss

from burhava import AssumptionGraph, Loss, Problem, marginal_value, neg, set_risk


def _two_atoms(rule: str) -> Problem:
    g = AssumptionGraph()
    g.add_atom("Z1", 0.5)  # perfect verifier: alpha = beta = 0 (defaults)
    g.add_atom("Z2", 0.5)
    if rule == "and":
        g.add_and("Y", ["Z1", "Z2"])
    else:  # XOR = (Z1 AND NOT Z2) OR (NOT Z1 AND Z2)
        g.add_derived("Y", [(["Z1", neg("Z2")],), ([neg("Z1"), "Z2"],)])
    g.set_conclusion("Y")
    return Problem(g)


@pytest.mark.parametrize(
    "rule, expected",
    [("and", (0.25, 0.25, 0.25, 0.0)), ("xor", (0.5, 0.5, 0.5, 0.0))],
)
def test_and_and_xor_counterexamples(rule, expected):
    b = _two_atoms(rule).prior_belief()
    L = Loss.zero_one()
    got = (set_risk(b, L, []), set_risk(b, L, [0]), set_risk(b, L, [1]), set_risk(b, L, [0, 1]))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_selection_example():
    g = AssumptionGraph()
    g.add_atom("Z1", 0.90)
    g.add_atom("Z2", 0.60)
    g.add_atom("Z3", 0.55)
    g.add_or("Z2orZ3", ["Z2", "Z3"])
    g.add_and("Y", ["Z1", "Z2orZ3"])
    g.set_conclusion("Y")
    b = Problem(g).prior_belief()
    L = Loss.from_costs(false_accept=10, false_reject=1, abstain=0.6)
    assert set_risk(b, L, []) == pytest.approx(0.6, abs=5e-4)
    deltas = [marginal_value(b, L, [], i) for i in range(3)]
    np.testing.assert_allclose(deltas, [0.060, 0.042, 0.027], atol=5e-4)


@settings(
    max_examples=500,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(seed=st.integers(0, 2**32 - 1))
def test_monotonicity(seed):
    """R(S + {i}) <= R(S) for random graphs, losses, verifier errors and evidence."""
    rng = np.random.default_rng(seed)
    prob = Problem(random_graph(rng))
    L = random_loss(rng)
    b = prob.prior_belief()
    order = rng.permutation(prob.n)
    # Some atoms already verified, verdicts drawn from the predictive P(O_i | E).
    n_seen = int(rng.integers(0, prob.n))
    for i in order[:n_seen]:
        b = b.observe(int(i), int(rng.random() < b.prob_verdict(int(i))))
    rest = [int(i) for i in order[n_seen:]]
    k = int(rng.integers(0, len(rest)))
    S, i = rest[:k], rest[k]
    r_s, r_si = set_risk(b, L, S), set_risk(b, L, S + [i])
    assert r_si <= r_s + 1e-9 * max(1.0, abs(r_s))
