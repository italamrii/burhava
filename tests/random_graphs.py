"""Random small graphs for property tests (test-only; not the M3 generators)."""

from __future__ import annotations

import numpy as np

from burhava.graph import AssumptionGraph, Derivation, Literal
from burhava.risk import Loss


def random_graph(rng: np.random.Generator, max_atoms: int = 7) -> AssumptionGraph:
    """A random AND/OR graph with random sources, step atoms, negations and
    verifier error rates anywhere in [0, 1]."""
    g = AssumptionGraph()
    n_src = int(rng.integers(0, 3))
    for k in range(n_src):
        m = int(rng.integers(2, 4))
        g.add_source(f"s{k}", rng.dirichlet(np.ones(m)))
    n = int(rng.integers(1, max_atoms + 1))
    for i in range(n):
        kind = "step" if rng.random() < 0.2 else "assumption"
        if n_src and rng.random() < 0.6:
            src = f"s{int(rng.integers(n_src))}"
            m = len(g.sources[src].u_probs)
            prior = rng.random(m)
        else:
            src, prior = None, float(rng.random())
        g.add_atom(
            f"Z{i}",
            prior,
            kind=kind,
            source=src,
            cost=float(rng.uniform(0.5, 5)),
            alpha=float(rng.random()),
            beta=float(rng.random()),
        )
    nodes = [a.name for a in g.atoms]
    steps = [a.name for a in g.atoms if a.kind == "step"]
    for j in range(int(rng.integers(1, 5))):
        ders = []
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(1, min(3, len(nodes)) + 1))
            prem = tuple(
                Literal(str(x), negated=bool(rng.random() < 0.15))
                for x in rng.choice(nodes, size=k, replace=False)
            )
            step = str(rng.choice(steps)) if steps and rng.random() < 0.3 else None
            ders.append(Derivation(prem, step))
        nodes.append(g.add_derived(f"D{j}", ders))
    g.set_conclusion(nodes[-1])
    return g


def random_loss(rng: np.random.Generator) -> Loss:
    """Any 3x2 loss matrix with entries in [0, 10] (not necessarily 'sensible')."""
    return Loss(rng.uniform(0, 10, size=(3, 2)))
