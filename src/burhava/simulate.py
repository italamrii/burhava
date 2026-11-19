"""Run one policy on one instance, with paired randomness.

Paired design: for a given ``seed`` the true state ``z``, every possible verifier
verdict and every self-consistency sample are drawn *before* any policy runs, so
all arms face exactly the same world and the same verifier answers. Arms differ
only in which checks they ask for.

Two problems are involved:

* ``truth``  - the real world: ``z`` is sampled from its joint, verdicts use its
  true error rates ``alpha, beta``, and ``Y`` is computed with its rules;
* ``model``  - what the method believes (equal to ``truth`` in Stage A; a
  corrupted copy in Stage B). Beliefs, risks and the final decision use it.

Both must have the same atoms in the same order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .model import Problem
from .policies import SAMPLE_Y, Policy, PolicyState, SelfConsistency, topo_order
from .risk import DECISIONS, Loss, bayes_decision, bayes_risk

MAX_SC_SAMPLES = 64


@dataclass(frozen=True)
class World:
    """Pre-drawn randomness for one instance (shared by all arms).

    Attributes:
        state: index of the true joint state ``z``.
        z: true atom values, bool (n,).
        y: true conclusion.
        verdicts: the verdict each atom WOULD return if checked, int (n,).
        y_samples: noisy conclusion readings for the self-consistency arm.
        seed: the seed that generated this world.
    """

    state: int
    z: np.ndarray
    y: bool
    verdicts: np.ndarray
    y_samples: np.ndarray
    seed: int


def draw_world(truth: Problem, seed: int, sc_eps: float = 0.3) -> World:
    """Sample the true state, all verdicts and SC samples from ``seed``."""
    rng = np.random.default_rng(seed)
    s = int(rng.choice(truth.joint.size, p=truth.joint / truth.joint.sum()))
    z = truth.bits[:, s].copy()
    u = rng.random(truth.n)
    # O_i = 1 if (Z_i = 1 and no false negative) or (Z_i = 0 and false positive)
    verdicts = np.where(z, u >= truth.beta, u < truth.alpha).astype(int)
    y = bool(truth.y[s])
    flips = rng.random(MAX_SC_SAMPLES) < sc_eps
    y_samples = np.where(flips, 1 - int(y), int(y)).astype(int)
    return World(s, z, y, verdicts, y_samples, seed)


def run_episode(
    policy: Policy,
    model: Problem,
    loss: Loss,
    world: World,
    budget: float,
    policy_seed: int = 0,
    true_loss: Loss | None = None,
) -> dict:
    """Run ``policy`` until it stops or the budget is spent; then decide.

    Args:
        policy: the arm.
        model: the problem the method believes in.
        loss: loss matrix used for deciding (and for scoring, unless
            ``true_loss`` is given).
        world: pre-drawn truth and verdicts (see :func:`draw_world`).
        budget: total verification cost allowed.
        policy_seed: seed for the policy's own randomness (arm 2).

    Returns:
        dict with decision, true y, realised loss, model risk at stop, final
        P(Y=1), checks used, cost spent, risk evaluations, wall time, and a
        flag for verdicts the model considered impossible.
    """
    t0 = time.perf_counter()
    state = PolicyState(
        belief=model.prior_belief(),
        loss=loss,
        rng=np.random.default_rng([policy_seed, 7]),
        topo_order=topo_order(model.graph),
    )
    spent, checks, n_sc, contradictions = 0.0, 0, 0, 0
    while True:
        left = budget - spent
        choice = policy.select(state, left)
        if choice is None:
            break
        if choice == SAMPLE_Y:
            assert isinstance(policy, SelfConsistency)
            if n_sc >= MAX_SC_SAMPLES:
                break
            o = int(world.y_samples[n_sc])
            n_sc += 1
            spent += policy.sample_cost
            state.belief = state.belief.observe_y(o, policy.eps)
        else:
            c = float(model.cost[choice])
            if c > left + 1e-9 or choice in state.belief.evidence:
                raise RuntimeError(f"{policy.name} chose an invalid atom {choice}")
            spent += c
            try:
                state.belief = state.belief.observe(choice, int(world.verdicts[choice]))
            except ValueError:
                # Verdict impossible under the (misspecified) model: record it as
                # checked with no information gained.
                contradictions += 1
                ev = {**state.belief.evidence, choice: int(world.verdicts[choice])}
                state.belief = type(state.belief)(
                    state.belief.problem, state.belief.w, ev, state.belief.y_obs
                )
        checks += 1
    b = state.belief
    d = bayes_decision(b, loss)
    scorer = true_loss or loss
    return {
        "arm": policy.name,
        "decision": DECISIONS[d],
        "y_true": int(world.y),
        "loss": float(scorer.matrix[d, int(world.y)]),
        "model_risk": bayes_risk(b, loss),
        "p_y": b.prob_y(),
        "checks": checks,
        "cost_spent": spent,
        "risk_evals": state.n_risk_evals,
        "contradictions": contradictions,
        "wall_time": time.perf_counter() - t0,
    }
