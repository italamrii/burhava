"""EXPLORATORY (not pre-registered): where does the advantage flip under edge errors?

Usage: python scripts/explore_edge.py results/stageB_edge
Prints the gain of arm 10 over arm 3 and over no verification (arm 1), per edge
error level, at verifier error 5% (matched) and ECE 0 / 0.10.
"""

import glob
import json
import sys

import numpy as np
import pandas as pd

from burhava.stats import paired_bootstrap

d = sys.argv[1]
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{d}/cell_*.csv.gz"))])
key = ["cell_id", "gid", "verifier", "assumption", "edge", "ece_target"]
w = df.pivot_table(index=key, columns="arm", values="loss", aggfunc="first").reset_index()
out = []
for t in sorted(w.ece_target.unique()):
    for e in sorted(w.edge.unique()):
        s = w[np.isclose(w.ece_target, t) & np.isclose(w.edge, e)]
        row = {"ece": t, "edge": e}
        for base in ("a03_lowconf", "a01_none", "a06_selfcons", "a08_propagation"):
            row[f"vs_{base}"] = paired_bootstrap((s[base] - s["a10_voi2"]).to_numpy())
        out.append(row)
        print(
            f"ECE {t:.2f} edge {e:.2f}: "
            + "  ".join(
                f"{k}={v['mean']:+.4f}[{v['lo']:+.4f},{v['hi']:+.4f}]"
                for k, v in row.items()
                if k.startswith("vs_")
            )
        )
json.dump(out, open("results/summaries/stageB_edge_explore.json", "w"), indent=2)
