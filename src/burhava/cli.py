"""Command line: ``burhava run | grid | report``.

* ``burhava run --seed 0``: one random instance, every arm, printed side by side.
* ``burhava grid --config configs/stageA.yaml --out results/stageA --n-graphs 1000``
* ``burhava report --stage A --in results/stageA``: hypothesis tests + figures.
"""

from __future__ import annotations

import argparse
import time


def _cmd_run(a):
    import numpy as np

    from .generators import GENERATORS, GenConfig
    from .model import Problem
    from .policies import all_arms
    from .risk import Loss
    from .simulate import draw_world, run_episode

    rng = np.random.default_rng(a.seed)
    g = GENERATORS[a.generator](GenConfig(structure=a.structure, depth=a.depth), rng)
    prob = Problem(g)
    world = draw_world(prob, a.seed)
    L = Loss.from_costs(10, 1, 0.6)
    print(f"{g.n} atoms, true Y = {int(world.y)}, P(Y=1) = {prob.joint[prob.y].sum():.3f}")
    for arm in all_arms():
        r = run_episode(arm, prob, L, world, a.budget, policy_seed=a.seed)
        print(
            f"{r['arm']:<18} decision={r['decision']:<8} loss={r['loss']:<5} "
            f"checks={r['checks']} model_risk={r['model_risk']:.3f}"
        )


def _cmd_grid(a):
    from .runner import load_config, run_grid

    t0 = time.perf_counter()
    cfg = load_config(a.config)
    if "noise" in cfg:
        from .stage_b import run_stage_b

        run_stage_b(cfg, a.out, a.n_graphs, a.workers)
    else:
        run_grid(cfg, a.out, a.n_graphs, a.workers)
    print(f"done in {time.perf_counter() - t0:.1f} s")


def _cmd_report(a):
    from .analysis import report

    report(a.stage, a.inp, a.figdir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="burhava")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one instance, all arms")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--generator", default="layered", choices=["layered", "rules"])
    r.add_argument("--structure", default="mixed")
    r.add_argument("--depth", type=int, default=4)
    r.add_argument("--budget", type=float, default=2)
    r.set_defaults(fn=_cmd_run)
    g = sub.add_parser("grid", help="run a grid from a YAML config")
    g.add_argument("--config", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n-graphs", type=int, required=True)
    g.add_argument("--workers", type=int, default=4)
    g.set_defaults(fn=_cmd_grid)
    rep = sub.add_parser("report", help="hypothesis tests and figures")
    rep.add_argument("--stage", required=True, choices=["A", "A-loss", "B"])
    rep.add_argument("--in", dest="inp", required=True)
    rep.add_argument("--figdir", default="results/figures")
    rep.set_defaults(fn=_cmd_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
