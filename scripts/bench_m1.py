"""M1 timing benchmark: exact risk evaluation at n = 20 atoms.

Usage:  python scripts/bench_m1.py --seed 0 [--out results/summaries/m1_timing.json]

Builds a random 20-atom graph (14 assumptions, 6 step atoms, 3 shared sources),
then times: building the joint table, rho, R(S) for |S| = 0..5, a full scan of
Delta_i(S) over all atoms with |S| = 2, and one Bayes update. Reports medians.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from burhava import AssumptionGraph, Loss, Problem, bayes_risk, marginal_value, set_risk


def build_graph(rng: np.random.Generator) -> AssumptionGraph:
    g = AssumptionGraph()
    for k in range(3):
        g.add_source(f"s{k}", [0.3, 0.7])
    names = []
    for i in range(14):
        src = f"s{i % 3}" if i < 9 else None
        prior = tuple(rng.uniform(0.3, 0.99, 2)) if src else float(rng.uniform(0.5, 0.99))
        names.append(g.add_atom(f"z{i}", prior, source=src, alpha=0.05, beta=0.05))
    steps = [g.add_step(f"t{j}", float(rng.uniform(0.8, 0.99))) for j in range(6)]
    layer = names
    j = 0
    while len(layer) > 1:  # alternate AND / OR layers over pairs
        nxt = []
        for a, b in zip(layer[0::2], layer[1::2], strict=False):
            name = f"d{j}"
            if j % 2 == 0:
                nxt.append(g.add_and(name, [a, b], step=steps[j % len(steps)]))
            else:
                nxt.append(g.add_or(name, [a, b]))
            j += 1
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    g.set_conclusion(layer[0])
    return g


def timeit(fn, repeat: int) -> float:
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    g = build_graph(rng)
    assert g.n == 20
    L = Loss.from_costs(10, 1, 0.6)
    res = {"n_atoms": g.n, "seed": args.seed, "repeat": args.repeat}
    res["build_problem_s"] = timeit(lambda: Problem(g), max(1, args.repeat // 2))
    prob = Problem(g)
    b = prob.prior_belief()
    res["rho_s"] = timeit(lambda: bayes_risk(b, L), args.repeat)
    for k in range(6):
        S = list(range(k))
        res[f"R_S_size{k}_s"] = timeit(lambda S=S: set_risk(b, L, S), args.repeat)
    res["delta_scan_all_atoms_S2_s"] = timeit(
        lambda: [marginal_value(b, L, [0, 1], i) for i in range(2, 20)], 3
    )
    res["bayes_update_s"] = timeit(lambda: b.observe(5, 1), args.repeat)
    res["machine"] = f"{platform.processor() or platform.machine()}, {os.cpu_count()} cpus"
    res["python"] = platform.python_version()
    res["numpy"] = np.__version__
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
