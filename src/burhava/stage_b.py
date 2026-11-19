"""Stage B (M4): noise robustness runner and report.

The *truth* is the generated graph with true verifier error ``e``. The *method*
sees a model with: logit-noise miscalibrated priors (scale chosen per target
ECE), corrupted dependency edges, and an assumed verifier error that is either
correct (matched) or wrong (mismatched). Random draws are shared across noise
levels (paired): the same ``eps`` vector scales with sigma, the same edge RNG
per edge level, and the same world uniforms for every verifier level.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .generators import GENERATORS, GenConfig
from .model import MixtureProblem, Problem
from .noise import (
    corrupt_edges,
    graph_ensemble,
    marginal_priors,
    perturb_calibration,
    population_ece,
    with_verifier,
)
from .policies import LookaheadVoI, arms_for
from .risk import Loss
from .runner import _graph_seeds, cells, config_hash
from .simulate import draw_world, run_episode
from .stats import paired_bootstrap


def assumed_error(true_e: float, assumption: str) -> float:
    if assumption == "matched":
        return true_e
    return 0.0 if true_e > 0 else 0.05


def run_cell_b(cell: dict, cfg: dict, out_dir: Path, n_graphs: int) -> Path:
    path = out_dir / f"cell_{cell['cell_id']:04d}.csv.gz"
    if path.exists():
        return path
    nz = cfg["noise"]
    arms = arms_for(cfg["arms"])
    rob = cfg.get("robust")
    robust_arm = LookaheadVoI(eta=cfg["arms"]["eta"])
    robust_arm.name = "a12_robust_voi"
    lc = cfg["loss"]
    L = Loss.from_costs(10 * lc["false_reject"], lc["false_reject"], lc["abstain"])
    budget = float(cfg["budgets"][0])
    gens = cfg["generators"]
    rows, cal_rows, edge_rows = [], [], []
    for gid in range(n_graphs):
        gen = gens[gid % len(gens)]
        grng, wseed = _graph_seeds(cfg["seed"], cell["cell_id"], gid)
        gc = GenConfig(
            cell["structure"],
            int(cell["depth"]),
            float(cell["false_rate"]),
            cell["correlation"],
            cell["cost"],
            cfg.get("max_atoms", 14),
        )
        graph = GENERATORS[gen](gc, grng)
        eps = grng.standard_normal(graph.n)
        p_true = marginal_priors(graph)
        edge_graphs = {}
        for k, r in enumerate(nz["edge"]):
            erng = np.random.default_rng([wseed, 101, k])
            eg, n_e, n_c = corrupt_edges(graph, float(r), erng)
            edge_graphs[r] = eg
            edge_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "gid": gid,
                    "edge": r,
                    "n_edges": n_e,
                    "n_changed": n_c,
                }
            )
        # miscalibrated graphs (structure from each edge level)
        mis = {}
        for (r, eg), t in itertools.product(edge_graphs.items(), nz["ece_targets"]):
            sigma = float(nz["ece_sigma"][t])
            mis[(r, t)] = perturb_calibration(eg, sigma, eps) if sigma > 0 else eg
            if r == nz["edge"][0]:
                pm = marginal_priors(mis[(r, t)])
                for i in range(graph.n):
                    cal_rows.append(
                        {
                            "cell_id": cell["cell_id"],
                            "gid": gid,
                            "ece_target": t,
                            "p_true": p_true[i],
                            "p_model": pm[i],
                        }
                    )
        for e in nz["verifier"]:
            truth = Problem(with_verifier(graph, float(e)))
            world = draw_world(truth, wseed, sc_eps=cfg["arms"]["sc_eps"])
            for assumption in nz["assumption"]:
                ae = assumed_error(float(e), assumption)
                for ci, ((r, t), mg) in enumerate(mis.items()):
                    model = Problem(with_verifier(mg, ae))
                    runs = [(a, model) for a in arms]
                    if rob:
                        ens_rng = np.random.default_rng([wseed, 202, ci])
                        ens = graph_ensemble(with_verifier(mg, ae), rob["r_hat"], rob["k"], ens_rng)
                        runs.append((robust_arm, MixtureProblem(ens)))
                    for arm, mdl in runs:
                        res = run_episode(arm, mdl, L, world, budget, policy_seed=wseed)
                        res.update(
                            cell_id=cell["cell_id"],
                            gid=gid,
                            gen=gen,
                            verifier=e,
                            assumption=assumption,
                            edge=r,
                            ece_target=t,
                        )
                        rows.append(res)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)
    pd.DataFrame(cal_rows).to_csv(out_dir / f"cal_{cell['cell_id']:04d}.csv.gz", index=False)
    pd.DataFrame(edge_rows).to_csv(out_dir / f"edge_{cell['cell_id']:04d}.csv.gz", index=False)
    return path


def _worker(args):
    cell, cfg, out_dir, n = args
    t0 = time.perf_counter()
    run_cell_b(cell, cfg, out_dir, n)
    return cell["cell_id"], time.perf_counter() - t0


def run_stage_b(cfg: dict, out_dir: str | Path, n_graphs: int, workers: int = 4) -> None:
    from multiprocessing import Pool

    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg, "config_hash": config_hash(cfg), "n_graphs": n_graphs}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    pd.DataFrame(cells(cfg)).to_csv(out_dir / "cells.csv", index=False)
    jobs = [(c, cfg, out_dir, n_graphs) for c in cells(cfg)]
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), mininterval=30):
            pass


# ------------------------------------------------------------------ report
NKEY = ["cell_id", "gid", "verifier", "assumption", "edge", "ece_target"]


def report_stage_b(inp: Path, figdir: Path) -> dict:
    from .analysis import fig_gain_lines

    df = pd.concat([pd.read_csv(p) for p in sorted(inp.glob("cell_*.csv.gz"))])
    df = df.merge(pd.read_csv(inp / "cells.csv"), on="cell_id")
    w = df.pivot_table(index=NKEY, columns="arm", values="loss", aggfunc="first").reset_index()
    w = w.merge(df.drop_duplicates(NKEY)[NKEY + ["structure", "gen"]], on=NKEY)

    def g(sub, base="a03_lowconf"):
        return paired_bootstrap((sub[base] - sub["a10_voi2"]).to_numpy())

    def sel(ece=0.0, v=0.0, a="matched", edge=0.0):
        return w[
            (np.isclose(w.ece_target, ece))
            & (np.isclose(w.verifier, v))
            & (w.assumption == a)
            & (np.isclose(w.edge, edge))
        ]

    out: dict = {"source": str(inp)}
    GA, GB = g(sel()), g(sel(ece=0.10, v=0.05))
    rej = bool(GB["mean"] < 0.25 * GA["mean"] or GB["lo"] <= 0)
    out["H4"] = {
        "G_A_clean": GA,
        "G_B_ece10_ver5_matched": GB,
        "ratio_GB_over_GA": GB["mean"] / GA["mean"] if GA["mean"] else None,
        "rejected": rej,
    }
    # calibration threshold X: largest ECE with gain CI excluding 0
    xs = {}
    for v in (0.0, 0.05, 0.15):
        ok = [t for t in (0.0, 0.05, 0.10, 0.20) if g(sel(ece=t, v=v))["lo"] > 0]
        xs[str(v)] = max(ok) if ok else None
    out["X_largest_ece_with_positive_gain"] = xs
    grid_rows = []
    for v, a, e, t in itertools.product(
        (0.0, 0.05, 0.15), ("matched", "mismatched"), (0.0, 0.10, 0.25), (0.0, 0.05, 0.10, 0.20)
    ):
        sub = sel(ece=t, v=v, a=a, edge=e)
        best = sub[["a03_lowconf", "a04_entropy", "a05_degree"]].mean().idxmin()
        grid_rows.append(
            {
                "verifier": v,
                "assumption": a,
                "edge": e,
                "ece": t,
                "vs_arm3": g(sub),
                "vs_best_3to5": g(sub, best),
                "best_3to5": best,
                "loss_arm10": float(sub["a10_voi2"].mean()),
                "loss_arm3": float(sub["a03_lowconf"].mean()),
            }
        )
    out["grid"] = grid_rows
    out["H4_by_structure"] = {
        s: g(sel(ece=0.10, v=0.05)[lambda x, s=s: x.structure == s])
        for s in ("chain", "and", "or", "mixed")
    }
    cal = pd.concat([pd.read_csv(p) for p in sorted(inp.glob("cal_*.csv.gz"))])
    out["achieved_ece"] = {
        str(t): population_ece(c.p_true.to_numpy(), c.p_model.to_numpy())
        for t, c in cal.groupby("ece_target")
    }
    ed = pd.concat([pd.read_csv(p) for p in sorted(inp.glob("edge_*.csv.gz"))])
    out["achieved_edge_error"] = {
        str(r): float(c.n_changed.sum() / c.n_edges.sum()) for r, c in ed.groupby("edge")
    }
    figdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"structure": lab, "ece": t, **g(sel(ece=t, v=v, a=a))}
        for lab, v, a in [
            ("chain", 0.0, "matched"),
            ("and", 0.05, "matched"),
            ("or", 0.15, "matched"),
            ("mixed", 0.15, "mismatched"),
        ]
        for t in (0.0, 0.05, 0.10, 0.20)
    ]
    labels = {
        "chain": "verifier 0%",
        "and": "verifier 5%",
        "or": "verifier 15%",
        "mixed": "verifier 15%, assumed 0%",
    }
    fig_gain_lines(
        rows,
        "ece",
        "calibration error of priors (target ECE)",
        "Stage B: arm 10 vs arm 3 under miscalibration",
        figdir / "stageB_gain_vs_ece.png",
        labels=labels,
    )
    rows_e = [
        {"structure": lab, "edge": e, **g(sel(ece=t, v=0.05, edge=e))}
        for lab, t in [("chain", 0.0), ("and", 0.10)]
        for e in (0.0, 0.10, 0.25)
    ]
    fig_gain_lines(
        rows_e,
        "edge",
        "dependency-edge error rate",
        "Stage B: arm 10 vs arm 3 under edge errors (verifier 5%)",
        figdir / "stageB_gain_vs_edge.png",
        labels={"chain": "ECE 0", "and": "ECE 0.10"},
    )
    return out
