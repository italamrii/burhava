"""Phase 2 hypothesis tests H6a-c and H7a-d, exactly as in docs/PREREGISTRATION_2.md.

Usage: python scripts/report_phase2.py
Reads results/phase2_A and results/phase2_B, writes
results/summaries/phase2_summary.json and two figures.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from burhava.analysis import fig_gain_lines
from burhava.stats import paired_bootstrap

OUT = Path("results/summaries/phase2_summary.json")
FIG = Path("results/figures")


def load(d):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{d}/cell_*.csv.gz"))])
    return df.merge(pd.read_csv(f"{d}/cells.csv"), on="cell_id")


def gain(w, base, arm):
    return paired_bootstrap((w[base] - w[arm]).to_numpy())


res = {}
# ---------------- P2-A
a = load("results/phase2_A")
key = ["cell_id", "gid", "budget", "ratio"]
wa = a.pivot_table(index=key, columns="arm", values="loss", aggfunc="first").reset_index()
wa = wa.merge(a.drop_duplicates(key)[key + ["structure"]], on=key)
b35 = wa[wa.budget.isin([3, 5])]
h6a = gain(b35, "a10_voi2", "a11_voi_spend")
h6b = gain(wa, "a08_propagation", "a11_voi_spend")
res["H6a_arm11_vs_arm10_b3_5"] = {**h6a, "rejected": not h6a["lo"] > 0}
res["H6b_arm11_noninferior_arm8"] = {**h6b, "rejected": h6b["lo"] < -0.002}
res["H6c_arm11_vs_arm8"] = {**h6b, "rejected": not h6b["lo"] > 0}
rows_a = []
for s in ["chain", "and", "or", "mixed"]:
    for bud in sorted(wa.budget.unique()):
        sub = wa[(wa.structure == s) & (wa.budget == bud)]
        rows_a.append(
            {
                "structure": s,
                "budget": bud,
                **gain(sub, "a08_propagation", "a11_voi_spend"),
                "vs_arm10": gain(sub, "a10_voi2", "a11_voi_spend"),
                "vs_arm3": gain(sub, "a03_lowconf", "a11_voi_spend"),
            }
        )
res["P2A_by_structure_budget"] = rows_a
res["P2A_mean_loss_by_budget"] = (
    a.groupby(["budget", "arm"]).loss.mean().unstack().round(4).to_dict(orient="index")
)

# ---------------- P2-B
b = load("results/phase2_B")
kb = ["cell_id", "gid", "verifier", "assumption", "edge", "ece_target"]
wb = b.pivot_table(index=kb, columns="arm", values="loss", aggfunc="first").reset_index()
wb = wb.merge(b.drop_duplicates(kb)[kb + ["structure"]], on=kb)


def sel(edge, ece=0.10):
    return wb[np.isclose(wb.edge, edge) & np.isclose(wb.ece_target, ece)]


h7a = gain(sel(0.10), "a10_voi2", "a12_robust_voi")
h7b = gain(sel(0.10), "a03_lowconf", "a12_robust_voi")
c = sel(0.0)
h7c = paired_bootstrap((c["a12_robust_voi"] - c["a10_voi2"]).to_numpy())
h7d = gain(sel(0.25), "a01_none", "a12_robust_voi")
res["H7a_arm12_vs_arm10_edge10"] = {**h7a, "rejected": not h7a["lo"] > 0}
res["H7b_arm12_vs_arm3_edge10"] = {**h7b, "rejected": not h7b["lo"] > 0}
res["H7c_cost_at_edge0"] = {**h7c, "margin": 0.0162, "rejected": h7c["mean"] > 0.0162}
res["H7d_arm12_vs_none_edge25"] = {**h7d, "rejected": not h7d["lo"] > 0}
rows_b, tab = [], []
for t in (0.0, 0.10):
    for e in (0.0, 0.05, 0.10, 0.25):
        s = sel(e, t)
        row = {
            "ece": t,
            "edge": e,
            "loss": {k: float(s[k].mean()) for k in s.columns if k.startswith(("a0", "a1"))},
        }
        for base in ("a03_lowconf", "a10_voi2", "a01_none", "a06_selfcons", "a08_propagation"):
            row[f"a12_vs_{base}"] = gain(s, base, "a12_robust_voi")
        row["a10_vs_a03"] = gain(s, "a03_lowconf", "a10_voi2")
        tab.append(row)
        if t == 0.10:
            rows_b.append({"structure": "chain", "edge": e, **row["a10_vs_a03"]})
            rows_b.append({"structure": "and", "edge": e, **row["a12_vs_a03_lowconf"]})
res["P2B_table"] = tab
res["P2B_H7a_by_structure"] = {
    s: gain(sel(0.10)[sel(0.10).structure == s], "a10_voi2", "a12_robust_voi")
    for s in ["chain", "and", "or", "mixed"]
}
OUT.write_text(json.dumps(res, indent=2, default=float))

fig_gain_lines(
    rows_b,
    "edge",
    "true dependency-edge error rate",
    "Robust VoI vs plain VoI: gain over arm 3 (ECE 0.10, verifier 5%)",
    FIG / "phase2_robust_vs_edge.png",
    labels={"chain": "arm 10 (plain VoI)", "and": "arm 12 (robust VoI)"},
    ylabel="loss reduction vs. arm 3\n(positive = named arm better)",
)
rows_s = [
    {
        "structure": r["structure"],
        "budget": r["budget"],
        "mean": r["mean"],
        "lo": r["lo"],
        "hi": r["hi"],
    }
    for r in rows_a
]
fig_gain_lines(
    rows_s,
    "budget",
    "verification budget (cost units)",
    "Arm 11 (VoI then spend) vs arm 8 (loss-blind influence)",
    FIG / "phase2_arm11_vs_arm8.png",
    ylabel="loss reduction vs. arm 8\n(positive = arm 11 better)",
)
for k, v in res.items():
    if k.startswith("H"):
        print(
            k,
            f"{v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]",
            "REJECTED" if v["rejected"] else "supported",
        )
