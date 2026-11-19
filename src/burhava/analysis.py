"""Hypothesis tests, summary tables and figures (``burhava report``).

Every number in docs/RESULTS.md is produced here from files in ``results/``.
Gains are ``loss(baseline) - loss(arm 10)`` per instance: positive = arm 10 better.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import load_results, load_stats
from .stats import ece, paired_bootstrap

STRUCTS = ["chain", "and", "or", "mixed"]
# categorical slots 1-4 of the reference palette (fixed order, never cycled)
COLORS = {"chain": "#2a78d6", "and": "#eb6834", "or": "#1baf7a", "mixed": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KEY = ["cell_id", "gid", "budget", "ratio"]
HEURISTICS = ["a03_lowconf", "a04_entropy", "a05_degree", "a08_propagation"]


def wide(df: pd.DataFrame, col: str = "loss") -> pd.DataFrame:
    """One row per instance x budget x ratio, one column per arm."""
    w = df.pivot_table(index=KEY, columns="arm", values=col, aggfunc="first")
    facts = df.drop_duplicates(KEY).set_index(KEY)
    keep = [
        c for c in ["gen", "structure", "depth", "false_rate", "correlation", "cost"] if c in facts
    ]
    return w.join(facts[keep]).reset_index()


def gain(w: pd.DataFrame, base: str, arm: str = "a10_voi2") -> dict:
    return paired_bootstrap((w[base] - w[arm]).to_numpy())


def _fmt(r: dict) -> str:
    return f"{r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] (n={r['n']})"


def arm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Secondary metrics per arm (and budget)."""
    d = df.copy()
    d["error"] = ((d.decision == "accept") & (d.y_true == 0)) | (
        (d.decision == "reject") & (d.y_true == 1)
    )
    d["abstain"] = d.decision == "abstain"
    d["brier"] = (d.p_y - d.y_true) ** 2
    prior = d[d.arm == "a01_none"].set_index(KEY)["model_risk"].rename("prior_risk")
    d = d.join(prior, on=KEY)
    d["risk_red_per_cost"] = np.where(
        d.cost_spent > 0, (d.prior_risk - d.model_risk) / d.cost_spent.clip(lower=1e-12), np.nan
    )
    g = d.groupby(["budget", "arm"])
    out = g.agg(
        loss=("loss", "mean"),
        error_rate=("error", "mean"),
        abstain_rate=("abstain", "mean"),
        brier=("brier", "mean"),
        model_risk=("model_risk", "mean"),
        risk_red_per_cost=("risk_red_per_cost", "mean"),
        checks=("checks", "mean"),
        risk_evals=("risk_evals", "mean"),
        ms_per_episode=("wall_time", lambda x: 1000 * x.mean()),
    )
    out["ece_py"] = g.apply(lambda x: ece(x.p_y, x.y_true), include_groups=False)
    return out.round(4)


# ------------------------------------------------------------------ figures
def _style(ax, zero_line=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0, color=INK2, linewidth=1)


def fig_gain_lines(
    rows: list[dict],
    xkey: str,
    xlabel: str,
    title: str,
    path: Path,
    logx=False,
    labels=None,
    ylabel="loss reduction vs. baseline\n(positive = arm 10 better)",
):
    """One line per structure: gain of arm 10 over a baseline vs ``xkey``, with 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    xs = sorted(df[xkey].unique())
    ends = []
    for k, s in enumerate(STRUCTS):
        d = df[df.structure == s].sort_values(xkey)
        if d.empty:
            continue
        # small horizontal dodge so CIs do not overlap: 1.5% of the x-range per series
        span = (max(xs) - min(xs)) or 1.0
        off = (k - 1.5) * (0.04 if logx else 0.015 * span)
        x = d[xkey].to_numpy(float) * (10**off if logx else 1) + (0 if logx else off)
        ax.errorbar(
            x,
            d["mean"],
            yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]],
            color=COLORS[s],
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=0,
            elinewidth=1.2,
            label=(labels or {}).get(s, s),
        )
        ends.append([d["mean"].iloc[-1], (labels or {}).get(s, s), x[-1]])
    # end-of-line labels, pushed apart so they never collide
    lo_y, hi_y = ax.get_ylim()
    gap = 0.07 * (hi_y - lo_y)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, s, xv in ends:
        ax.annotate(
            s,
            (xv, yv),
            xytext=(8, 0),
            textcoords="offset points",
            color=INK2,
            fontsize=9,
            va="center",
        )
    if logx:
        ax.set_xscale("log")
    ax.margins(x=0.1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_arm_dots(tab: pd.DataFrame, title: str, path: Path):
    """Mean loss per arm (single series, sorted), with 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tab.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200, facecolor=SURFACE)
    _style(ax, zero_line=False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(t))
    ax.errorbar(
        t["mean"],
        y,
        xerr=[t["mean"] - t["lo"], t["hi"] - t["mean"]],
        fmt="o",
        color="#2a78d6",
        markersize=6,
        elinewidth=1.2,
        capsize=0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, color=INK2, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean realised decision loss (lower is better)", color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ stage A
def report_stage_a(inp: Path, figdir: Path) -> dict:
    df = load_results(inp)
    df = df[df.ratio == 10]
    w = wide(df)
    out: dict = {"source": str(inp), "n_rows": int(len(df))}

    # H1
    h1 = {}
    for s in STRUCTS:
        r = gain(w[(w.structure == s) & (w.budget == 2)], "a03_lowconf")
        r["fails"] = bool(r["lo"] <= 0)
        h1[s] = r
    n_fail = sum(r["fails"] for r in h1.values())
    out["H1"] = {"per_structure": h1, "n_fail": n_fail, "rejected": n_fail >= 3}
    out["H1_by_generator"] = {
        g: {
            s: gain(w[(w.structure == s) & (w.budget == 2) & (w.gen == g)], "a03_lowconf")
            for s in STRUCTS
        }
        for g in sorted(w.gen.unique())
    }
    out["H1_by_cost"] = {
        c: {
            s: gain(w[(w.structure == s) & (w.budget == 2) & (w.cost == c)], "a03_lowconf")
            for s in STRUCTS
        }
        for c in ["uniform", "varied"]
    }
    # H2
    r2 = gain(w[(w.structure == "and") & w.budget.isin([2, 3])], "a09_voi1")
    out["H2"] = {**r2, "rejected": bool(r2["lo"] <= 0)}
    # H3
    wc = w[w.correlation == "shared"]
    D = paired_bootstrap((wc["a10_voi2_indep"] - wc["a10_voi2"]).to_numpy())
    G = gain(wc, "a03_lowconf")
    out["H3"] = {
        "D_indep_minus_full": D,
        "G_h1_gain_corr_cells": G,
        "ratio_absD_over_G": abs(D["mean"]) / G["mean"] if G["mean"] > 0 else None,
        "rejected": (abs(D["mean"]) < 0.05 * G["mean"]) if G["mean"] > 0 else None,
        "evaluable": G["mean"] > 0,
    }
    # gains vs other baselines, all budgets
    rows, rows_best = [], []
    for s in STRUCTS:
        for b in sorted(w.budget.unique()):
            sub = w[(w.structure == s) & (w.budget == b)]
            rows.append({"structure": s, "budget": b, **gain(sub, "a03_lowconf")})
            best = (
                sub[["a03_lowconf", "a04_entropy", "a05_degree", "a08_propagation"]].mean().idxmin()
            )
            rows_best.append(
                {"structure": s, "budget": b, "best_heuristic": best, **gain(sub, best)}
            )
    out["gain_vs_arm3_by_structure_budget"] = rows
    out["gain_vs_best_heuristic_by_structure_budget"] = rows_best
    out["gain_vs_arm3_by_depth_b2"] = [
        {"depth": int(d), **gain(w[(w.depth == d) & (w.budget == 2)], "a03_lowconf")}
        for d in sorted(w.depth.unique())
    ]
    out["gain_vs_arm3_by_false_rate_b2"] = [
        {"false_rate": float(f), **gain(w[(w.false_rate == f) & (w.budget == 2)], "a03_lowconf")}
        for f in sorted(w.false_rate.unique())
    ]
    # arm-level tables
    tab_b2 = {}
    sub = w[w.budget == 2]
    for arm in [c for c in w.columns if c.startswith("a")]:
        if arm in sub and sub[arm].notna().any():
            tab_b2[arm] = paired_bootstrap(sub[arm].dropna().to_numpy())
    out["mean_loss_b2"] = tab_b2
    at = arm_table(df)
    at.to_csv(figdir.parent / "summaries" / "stageA_arm_table.csv")
    st = load_stats(inp)
    sst = (
        st.groupby(["gen", "structure"])[
            [
                "n_atoms",
                "n_steps",
                "n_derived",
                "n_relevant",
                "p_y",
                "mean_false_prob",
                "mean_degree",
            ]
        ]
        .mean()
        .round(3)
    )
    sst.to_csv(figdir.parent / "summaries" / "stageA_structure_stats.csv")

    figdir.mkdir(parents=True, exist_ok=True)
    fig_gain_lines(
        rows,
        "budget",
        "verification budget (cost units)",
        "Stage A: arm 10 vs lowest-confidence-first (arm 3)",
        figdir / "stageA_gain_vs_arm3.png",
    )
    fig_gain_lines(
        rows_best,
        "budget",
        "verification budget (cost units)",
        "Stage A: arm 10 vs best confidence/structure heuristic",
        figdir / "stageA_gain_vs_best_heuristic.png",
    )
    fig_arm_dots(
        pd.DataFrame(tab_b2).T,
        "Stage A, budget 2: mean loss by arm (95% CI)",
        figdir / "stageA_loss_by_arm_b2.png",
    )
    return out


def report_stage_a_loss(inp: Path, figdir: Path) -> dict:
    df = load_results(inp)
    w = wide(df)
    rows = []
    for s in STRUCTS:
        for r in sorted(w.ratio.unique()):
            rows.append(
                {
                    "structure": s,
                    "ratio": int(r),
                    **gain(w[(w.structure == s) & (w.ratio == r)], "a03_lowconf"),
                }
            )
    figdir.mkdir(parents=True, exist_ok=True)
    fig_gain_lines(
        rows,
        "ratio",
        "false-accept / false-reject loss ratio",
        "Loss sweep (budget 2): arm 10 vs arm 3",
        figdir / "stageA_loss_sweep.png",
        logx=True,
    )
    return {"source": str(inp), "gain_vs_arm3_by_structure_ratio": rows}


def report(stage: str, inp: str, figdir: str = "results/figures") -> None:
    inp_p, fig_p = Path(inp), Path(figdir)
    (fig_p.parent / "summaries").mkdir(parents=True, exist_ok=True)
    if stage == "A":
        out = report_stage_a(inp_p, fig_p)
        name = "stageA_summary.json"
    elif stage == "A-loss":
        out = report_stage_a_loss(inp_p, fig_p)
        name = "stageA_loss_summary.json"
    else:
        from .stage_b import report_stage_b

        out = report_stage_b(inp_p, fig_p)
        name = "stageB_summary.json"
    path = fig_p.parent / "summaries" / name
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {path}")
    for k in ("H1", "H2", "H3", "H4"):
        if k in out:
            print(k, json.dumps(out[k], default=float)[:600])
