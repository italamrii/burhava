import numpy as np
import pytest
from random_graphs import random_graph, random_loss

from burhava import AssumptionGraph, Loss, Problem
from burhava.policies import (
    SAMPLE_Y,
    HighestDegree,
    HighestEntropy,
    LookaheadVoI,
    LowestConfidence,
    NoVerification,
    OneStepVoI,
    PolicyState,
    PropagationScore,
    RandomPolicy,
    SelfConsistency,
    TopologicalStopAtFailure,
    all_arms,
    candidates,
    topo_order,
)
from burhava.simulate import draw_world, run_episode


def _state(prob, loss=None, belief=None):
    return PolicyState(
        belief=belief or prob.prior_belief(),
        loss=loss or Loss.zero_one(),
        rng=np.random.default_rng(0),
        topo_order=topo_order(prob.graph),
    )


def and_problem():
    g = AssumptionGraph()
    g.add_atom("Z1", 0.5)
    g.add_atom("Z2", 0.5)
    g.add_and("Y", ["Z1", "Z2"])
    g.set_conclusion("Y")
    return Problem(g)


def selection_problem():
    g = AssumptionGraph()
    g.add_atom("Z1", 0.90)
    g.add_atom("Z2", 0.60)
    g.add_atom("Z3", 0.55)
    g.add_or("o", ["Z2", "Z3"])
    g.add_and("Y", ["Z1", "o"])
    g.set_conclusion("Y")
    return Problem(g)


SEL_LOSS = Loss.from_costs(10, 1, 0.6)


# ---- the two M2 'done' tests ------------------------------------------------
def test_arm9_stalls_on_and_but_arm10_does_not():
    prob = and_problem()
    assert OneStepVoI().select(_state(prob), budget_left=2) is None
    assert LookaheadVoI().select(_state(prob), budget_left=2) == 0
    # with only one check left a pair is unaffordable, so arm 10 also stops
    assert LookaheadVoI().select(_state(prob), budget_left=1) is None
    # end to end: arm 10 always finds the truth with budget 2, arm 9 never checks
    for seed in range(20):
        w = draw_world(prob, seed)
        r10 = run_episode(LookaheadVoI(), prob, Loss.zero_one(), w, budget=2)
        r9 = run_episode(OneStepVoI(), prob, Loss.zero_one(), w, budget=2)
        assert r9["checks"] == 0
        assert r10["loss"] == 0.0
        assert r10["model_risk"] == pytest.approx(0.0)


def test_arms_9_and_10_pick_z1_on_selection_example():
    prob = selection_problem()
    assert OneStepVoI().select(_state(prob, SEL_LOSS), 1) == 0
    assert LookaheadVoI().select(_state(prob, SEL_LOSS), 3) == 0
    # whereas confidence-based arms pick Z3 (lowest confidence, highest entropy)
    assert LowestConfidence().select(_state(prob, SEL_LOSS), 1) == 2
    assert HighestEntropy().select(_state(prob, SEL_LOSS), 1) == 2


# ---- per-arm unit tests ------------------------------------------------------
def test_no_verification():
    assert NoVerification().select(_state(and_problem()), 5) is None


def test_random_returns_candidate_and_respects_budget():
    prob = selection_problem()
    st = _state(prob)
    picks = {RandomPolicy().select(st, 1) for _ in range(50)}
    assert picks == {0, 1, 2}
    assert RandomPolicy().select(st, 0.5) is None  # every cost is 1


def test_degree_prefers_shared_atom():
    g = AssumptionGraph()
    g.add_atom("a", 0.9)
    g.add_atom("b", 0.9)
    g.add_and("x", ["a", "b"])
    g.add_or("y", ["b", "x"])
    g.set_conclusion("y")
    assert HighestDegree().select(_state(Problem(g)), 1) == 1


def test_self_consistency():
    sc = SelfConsistency(eps=0.3, sample_cost=1.0)
    assert sc.select(_state(and_problem()), 1) == SAMPLE_Y
    assert sc.select(_state(and_problem()), 0.5) is None
    prob = and_problem()
    w = draw_world(prob, 3, sc_eps=0.3)
    r = run_episode(sc, prob, Loss.zero_one(), w, budget=3)
    assert r["checks"] == 3 and r["cost_spent"] == 3


def test_topological_stops_at_first_failure():
    prob = selection_problem()
    assert topo_order(prob.graph) == [1, 2, 0]  # o = OR(Z2, Z3) is defined first
    st = _state(prob)
    assert TopologicalStopAtFailure().select(st, 3) == 1
    st.belief = st.belief.observe(1, 0)
    assert TopologicalStopAtFailure().select(st, 3) is None


def test_propagation_score_matches_definition():
    prob = selection_problem()
    st = _state(prob)
    b = st.belief
    scores = []
    for i in range(3):
        p = b.prob_atom(i)
        py1 = b.observe(i, 1).prob_y()
        py0 = b.observe(i, 0).prob_y()
        scores.append(2 * p * (1 - p) * abs(py1 - py0))
    assert PropagationScore().select(st, 1) == int(np.argmax(scores))


def test_ties_lower_cost_then_lower_index():
    g = AssumptionGraph()
    g.add_atom("a", 0.5, cost=2.0)
    g.add_atom("b", 0.5, cost=1.0)
    g.add_atom("c", 0.5, cost=1.0)
    g.add_and("y", ["a", "b", "c"])
    g.set_conclusion("y")
    st = _state(Problem(g))
    assert LowestConfidence().select(st, 5) == 1


def test_all_arms_valid_on_random_instances():
    rng = np.random.default_rng(5)
    for t in range(40):
        prob = Problem(random_graph(rng))
        L = random_loss(rng)
        w = draw_world(prob, t)
        budget = float(rng.integers(0, 6))
        for arm in all_arms():
            r = run_episode(arm, prob, L, w, budget, policy_seed=t)
            assert r["cost_spent"] <= budget + 1e-9
            assert r["loss"] in set(L.matrix[:, int(w.y)].tolist())
            if arm.name != "a06_selfcons":
                assert r["checks"] <= len(candidates(_state(prob), np.inf))


def test_paired_world_is_deterministic():
    prob = selection_problem()
    a, b = draw_world(prob, 11), draw_world(prob, 11)
    assert a.state == b.state and (a.verdicts == b.verdicts).all()
