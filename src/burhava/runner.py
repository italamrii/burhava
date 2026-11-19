"""Grid runner shared by Stage A (M3) and Stage B (M4).

One *cell* = one combination of grid factors. For each cell, ``n_graphs`` graphs
are generated (alternating generator A / B), a world is drawn per graph, and
every arm runs at every budget and loss ratio on that same world. Results go
to one gzipped CSV per cell, so a run can be resumed and parallelised.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .generators import GENERATORS, GenConfig, structure_stats
from .model import Problem
from .noise import independent_version
from .policies import LookaheadVoI, arms_for
from .risk import Loss
from .simulate import draw_world, run_episode

FACTORS = ("structure", "depth", "false_rate", "correlation", "cost")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Short stable hash of a config (logged in EXPERIMENTS.md)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def cells(cfg: dict) -> list[dict]:
    """All grid cells, with a stable integer ``cell_id``."""
    grid = cfg["grid"]
    out = []
    for k, vals in enumerate(itertools.product(*[grid[f] for f in FACTORS])):
        out.append({"cell_id": k, **dict(zip(FACTORS, vals, strict=True))})
    return out


def _graph_seeds(base: int, cell_id: int, gid: int) -> tuple[np.random.Generator, int]:
    ss = np.random.SeedSequence([base, cell_id, gid])
    g_ss, w_ss = ss.spawn(2)
    return np.random.default_rng(g_ss), int(w_ss.generate_state(1)[0])


def run_cell(cell: dict, cfg: dict, out_dir: str | Path, n_graphs: int) -> Path:
    """Run every arm on ``n_graphs`` graphs of one cell; write CSV; return path."""
    out_dir = Path(out_dir)
    path = out_dir / f"cell_{cell['cell_id']:04d}.csv.gz"
    if path.exists():
        return path
    arms = arms_for(cfg["arms"])
    indep_arm = LookaheadVoI(eta=cfg["arms"]["eta"])
    indep_arm.name = "a10_voi2_indep"
    lc = cfg["loss"]
    losses = {
        r: Loss.from_costs(r * lc["false_reject"], lc["false_reject"], lc["abstain"])
        for r in lc["ratios"]
    }
    gens = cfg["generators"]
    rows, srows = [], []
    for gid in range(n_graphs):
        gen = gens[gid % len(gens)]
        grng, wseed = _graph_seeds(cfg["seed"], cell["cell_id"], gid)
        gc = GenConfig(
            structure=cell["structure"],
            depth=int(cell["depth"]),
            false_rate=float(cell["false_rate"]),
            correlation=cell["correlation"],
            cost=cell["cost"],
            max_atoms=cfg.get("max_atoms", 14),
        )
        graph = GENERATORS[gen](gc, grng)
        truth = Problem(graph)
        world = draw_world(truth, wseed, sc_eps=cfg["arms"]["sc_eps"])
        indep = Problem(independent_version(graph)) if cell["correlation"] == "shared" else None
        srows.append({"cell_id": cell["cell_id"], "gid": gid, "gen": gen, **structure_stats(graph)})
        for budget in cfg["budgets"]:
            for ratio, L in losses.items():
                runs = [(a, truth) for a in arms]
                if indep is not None and cfg["arms"].get("indep_arm", True):
                    runs.append((indep_arm, indep))
                for arm, model in runs:
                    r = run_episode(arm, model, L, world, float(budget), policy_seed=wseed)
                    r.update(cell_id=cell["cell_id"], gid=gid, gen=gen, budget=budget, ratio=ratio)
                    rows.append(r)
    df = pd.DataFrame(rows)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.rename(path)
    pd.DataFrame(srows).to_csv(out_dir / f"stats_{cell['cell_id']:04d}.csv.gz", index=False)
    return path


def _worker(args):
    cell, cfg, out_dir, n = args
    t0 = time.perf_counter()
    run_cell(cell, cfg, out_dir, n)
    return cell["cell_id"], time.perf_counter() - t0


def run_grid(cfg: dict, out_dir: str | Path, n_graphs: int, workers: int = 4) -> None:
    """Run all cells (skipping finished ones) with a process pool."""
    from multiprocessing import Pool

    from tqdm import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"config": cfg, "config_hash": config_hash(cfg), "n_graphs": n_graphs}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(cells(cfg)).to_csv(out_dir / "cells.csv", index=False)
    jobs = [(c, cfg, out_dir, n_graphs) for c in cells(cfg)]
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), mininterval=30):
            pass


def load_results(out_dir: str | Path) -> pd.DataFrame:
    """All per-episode rows joined with their cell factors."""
    out_dir = Path(out_dir)
    df = pd.concat(
        [pd.read_csv(p) for p in sorted(out_dir.glob("cell_*.csv.gz"))], ignore_index=True
    )
    return df.merge(pd.read_csv(out_dir / "cells.csv"), on="cell_id")


def load_stats(out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    df = pd.concat(
        [pd.read_csv(p) for p in sorted(out_dir.glob("stats_*.csv.gz"))], ignore_index=True
    )
    return df.merge(pd.read_csv(out_dir / "cells.csv"), on="cell_id")
