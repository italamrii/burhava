"""Paired bootstrap confidence intervals, effect sizes and calibration error."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(diff, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap CI for the mean of paired differences.

    Args:
        diff: per-instance differences ``loss(B) - loss(A)`` (pairing is already
            done: one number per instance).
        n_boot: number of resamples (10,000 per the pre-registration).
        seed: RNG seed.
        level: confidence level.

    Returns:
        ``{"mean", "lo", "hi", "n", "dz"}``. ``dz`` is Cohen's d for paired data
        (mean difference / standard deviation of differences).
    """
    d = np.asarray(diff, dtype=float)
    n = d.size
    if n == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "dz": np.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, int(4_000_000 // n))
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        means[s : s + k] = d[rng.integers(0, n, size=(k, n))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    sd = d.std(ddof=1) if n > 1 else np.nan
    return {
        "mean": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(n),
        "dz": float(d.mean() / sd) if sd and sd > 0 else 0.0,
    }


def ece(p, y, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins.

    ``ECE = sum_b (n_b / n) * |mean(y in b) - mean(p in b)|``.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / p.size)
