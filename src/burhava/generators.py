"""Two independent synthetic graph generators for Stage A.

Generator A, ``layered``: random layered AND/OR DAGs.
Generator B, ``rules``: ProofWriter-style theories. Facts, plus rules whose
bodies lead to a query through a proof of a given depth, plus alternative
proofs and distractor facts that are irrelevant to the query.

The two share only :class:`AtomFactory`, which turns the experimental
parameters (false-rate, correlation, cost) into atom priors and costs. They
build structure in deliberately different ways. Any finding that holds for
only one generator is reported as such.

How the experimental factors map onto atoms
-------------------------------------------
* **false-rate f**: each atom is false with prior probability ``f * g_i``,
  where ``g_i ~ Uniform(0.5, 1.5)`` spreads confidences around ``f``. Priors are
  clipped to ``[0.001, 0.999]``, so ``f = 0`` means "almost surely true".
* **correlation**: ``independent``, or ``shared`` with 2-3 sources. Under
  ``shared``, each assumption cites a random source ``k``, and the source is
  "bad" with ``P(U_k = bad) = 0.3``. A bad source makes its atoms 4x as likely
  to be false as a good one. The factors are chosen so that the *marginal*
  false probability is still ``f * g_i``:
  ``0.7 * a + 0.3 * 4a = 1`` gives ``a = 1/1.9``.
  Step atoms are always independent.
* **cost**: ``uniform`` (every ``c_i = 1``), or ``varied`` (integers 1-5,
  rescaled so the instance's mean cost is 1). The budget is in these units.
* Stage A verifier: perfect (``alpha = beta = 0``) unless ``verifier_error``
  is set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .graph import AssumptionGraph, Derivation, Literal, rebuild

STRUCTURES = ("chain", "and", "or", "mixed")
P_BAD = 0.3
BAD_RATIO = 4.0
A_GOOD = 1.0 / (1.0 - P_BAD + P_BAD * BAD_RATIO)  # = 1/1.9


@dataclass(frozen=True)
class GenConfig:
    """Parameters of one Stage A cell (see module docstring)."""

    structure: str = "mixed"
    depth: int = 4
    false_rate: float = 0.2
    correlation: str = "independent"  # or "shared"
    cost: str = "uniform"  # or "varied"
    max_atoms: int = 14
    verifier_error: float = 0.0


def _clip(p: float) -> float:
    return float(min(max(p, 0.001), 0.999))


class AtomFactory:
    """Creates atoms with priors/costs that follow a :class:`GenConfig`."""

    def __init__(self, g: AssumptionGraph, cfg: GenConfig, rng: np.random.Generator):
        self.g, self.cfg, self.rng = g, cfg, rng
        self.assumptions: list[str] = []
        self.n_sources = 0
        if cfg.correlation == "shared":
            self.n_sources = int(rng.integers(2, 4))
            for k in range(self.n_sources):
                g.add_source(f"src{k}", [1 - P_BAD, P_BAD])  # u=0 good, u=1 bad
        elif cfg.correlation != "independent":
            raise ValueError(f"unknown correlation {cfg.correlation!r}")

    def room(self, k: int = 1) -> bool:
        return self.g.n + k <= self.cfg.max_atoms

    def _common(self) -> dict:
        e = self.cfg.verifier_error
        cost = 1.0 if self.cfg.cost == "uniform" else float(self.rng.integers(1, 6))
        return {"cost": cost, "alpha": e, "beta": e}

    def assumption(self, allow_reuse: bool = True) -> str:
        """A fresh assumption atom, or an existing one if the atom cap is hit."""
        if not self.room():
            if not (allow_reuse and self.assumptions):
                raise RuntimeError("atom cap reached and nothing to reuse")
            return str(self.rng.choice(self.assumptions))
        f, gi = self.cfg.false_rate, self.rng.uniform(0.5, 1.5)
        name = f"A{len(self.assumptions)}"
        if self.n_sources:
            src = f"src{int(self.rng.integers(self.n_sources))}"
            prior = (_clip(1 - f * gi * A_GOOD), _clip(1 - f * gi * A_GOOD * BAD_RATIO))
            self.g.add_atom(name, prior, source=src, **self._common())
        else:
            self.g.add_atom(name, _clip(1 - f * gi), **self._common())
        self.assumptions.append(name)
        return name

    def maybe_step(self, prob: float) -> str | None:
        """With probability ``prob`` (and room under the cap) a new step atom."""
        if not self.room() or self.rng.random() >= prob:
            return None
        f, gi = self.cfg.false_rate, self.rng.uniform(0.5, 1.5)
        return self.g.add_step(f"T{self.g.n}", _clip(1 - f * gi), **self._common())

    def finish(self) -> AssumptionGraph:
        """Rescale varied costs to mean 1 and return the graph."""
        if self.cfg.cost == "uniform":
            return self.g
        costs = np.array([a.cost for a in self.g.atoms])
        atoms = [replace(a, cost=float(a.cost / costs.mean())) for a in self.g.atoms]
        return rebuild(self.g, atoms=atoms)


# ---------------------------------------------------------------- generator A
def layered_graph(cfg: GenConfig, rng: np.random.Generator) -> AssumptionGraph:
    """Generator A: a layered DAG with ``cfg.depth`` derivation layers.

    * chain: ``N_k = AND(N_{k-1}, A_k) [AND step_k]``, one node per layer.
    * and / or / mixed: up to 2 nodes per layer. Each takes 1-2 nodes of the
      layer below, plus (often) an assumption. The node joins its inputs with
      AND (one derivation), with OR (one derivation per input), or with a
      random choice of the two (mixed). Steps attach to derivations with
      probability 0.3.
    """
    if cfg.structure not in STRUCTURES:
        raise ValueError(f"unknown structure {cfg.structure!r}")
    g = AssumptionGraph()
    fac = AtomFactory(g, cfg, rng)
    if cfg.structure == "chain":
        prev = fac.assumption()
        for k in range(cfg.depth):
            step = fac.maybe_step(0.5)
            prem = [prev, fac.assumption()]
            prev = g.add_and(f"N{k}", prem, step=step)
        g.set_conclusion(prev)
        return fac.finish()

    layer = [fac.assumption() for _ in range(3)]
    for lvl in range(1, cfg.depth + 1):
        top = lvl == cfg.depth
        width = 1 if top else min(2, len(layer))
        new = []
        for k in range(width):
            if top:
                inputs = list(layer)
            else:
                inputs = [layer[k % len(layer)]]
                if len(layer) > 1 and rng.random() < 0.5:
                    inputs.append(layer[(k + 1) % len(layer)])
            if len(inputs) < 2 or rng.random() < 0.6:
                a = fac.assumption()
                if a not in inputs:
                    inputs.append(a)
            if len(inputs) < 2:  # reuse produced a duplicate: take any other atom
                others = [x for x in fac.assumptions if x not in inputs]
                inputs.append(str(rng.choice(others)))
            op = cfg.structure if cfg.structure != "mixed" else rng.choice(["and", "or"])
            name = f"L{lvl}_{k}"
            if op == "and":
                g.add_derived(name, [Derivation(tuple(map(Literal, inputs)), fac.maybe_step(0.3))])
            else:
                g.add_derived(
                    name, [Derivation((Literal(x),), fac.maybe_step(0.3)) for x in inputs]
                )
            new.append(name)
        layer = new
    g.set_conclusion(layer[0])
    return fac.finish()


# ---------------------------------------------------------------- generator B
def rules_graph(cfg: GenConfig, rng: np.random.Generator) -> AssumptionGraph:
    """Generator B: a ProofWriter-style theory.

    The query ``G_d`` has a proof of depth ``d``: ``G_0`` is a fact, and rule
    ``r_k: G_{k-1} AND body_k -> G_k``. Rules may be uncertain; a rule's
    validity is a step atom. Structure controls the rule bodies:

    * chain: body has no extra facts; rule-validity atoms with probability 0.5.
    * and: 1-2 extra facts per body.
    * or: 0-1 extra facts, plus with probability 0.7 an alternative rule
      ``fact -> G_k`` (a second proof route).
    * mixed: each level is drawn at random as one of the above.

    Then 1-2 *distractor* facts (true or false, never used by any rule leading
    to the query) are added, as in ProofWriter theories.
    """
    if cfg.structure not in STRUCTURES:
        raise ValueError(f"unknown structure {cfg.structure!r}")
    g = AssumptionGraph()
    fac = AtomFactory(g, cfg, rng)
    n_distract = int(rng.integers(1, 3))
    reserve = n_distract  # keep room for distractors
    goal = fac.assumption()  # G_0 is a fact
    for k in range(1, cfg.depth + 1):
        kind = cfg.structure if cfg.structure != "mixed" else rng.choice(["chain", "and", "or"])
        body: list[str] = [goal]
        if kind == "and":
            for _ in range(int(rng.integers(1, 3))):
                body.append(
                    fac.assumption() if fac.room(1 + reserve) else str(rng.choice(fac.assumptions))
                )
        elif kind == "or" and rng.random() < 0.5:
            body.append(
                fac.assumption() if fac.room(1 + reserve) else str(rng.choice(fac.assumptions))
            )
        body = list(dict.fromkeys(body))
        p_rule = 0.5 if kind == "chain" else 0.25
        rule = fac.maybe_step(p_rule) if fac.room(1 + reserve) else None
        ders = [Derivation(tuple(map(Literal, body)), rule)]
        if kind == "or" and rng.random() < 0.7:
            alt = fac.assumption() if fac.room(1 + reserve) else str(rng.choice(fac.assumptions))
            if alt not in body:
                ders.append(Derivation((Literal(alt),), None))
        goal = g.add_derived(f"G{k}", ders)
    g.set_conclusion(goal)
    for _ in range(n_distract):
        if fac.room():
            fac.assumption(allow_reuse=False)
    return fac.finish()


GENERATORS = {"layered": layered_graph, "rules": rules_graph}


def structure_stats(graph: AssumptionGraph) -> dict:
    """Descriptive statistics of one generated instance."""
    from .model import Problem

    p = Problem(graph)
    y = p.y
    idx = np.arange(y.size)
    relevant = sum(bool((y != y[idx ^ (1 << i)]).any()) for i in range(p.n))
    marg = p.prior_belief().prob_atoms()
    return {
        "n_atoms": p.n,
        "n_steps": sum(a.kind == "step" for a in graph.atoms),
        "n_derived": len(graph.derived),
        "n_relevant": relevant,
        "p_y": float(p.joint[y].sum()),
        "mean_false_prob": float((1 - marg).mean()),
        "mean_degree": float(graph.degree().mean()),
    }
