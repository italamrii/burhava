"""Find the logit-noise scale sigma that gives each target ECE (Stage B).

Usage: python scripts/calibrate_ece.py --seed 999
Samples 100 graphs per (generator, structure, false-rate, correlation) of the
Stage B sub-grid (depth 4, uniform cost), then bisects sigma so the population
ECE of the perturbed marginal priors equals each target. The printed values are
copied into configs/stageB.yaml.
"""

import argparse
import itertools

import numpy as np

from burhava.generators import GENERATORS, STRUCTURES, GenConfig
from burhava.noise import marginal_priors, sigma_for_ece

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=999)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
P, E = [], []
for s, f, c in itertools.product(STRUCTURES, [0.1, 0.2, 0.3], ["independent", "shared"]):
    for gen in GENERATORS:
        for _ in range(100):
            g = GENERATORS[gen](GenConfig(s, 4, f, c, "uniform"), rng)
            P.append(marginal_priors(g))
            E.append(rng.standard_normal(g.n))
P, E = np.concatenate(P), np.concatenate(E)
print(f"{P.size} atoms")
for t in (0.05, 0.10, 0.20):
    print(f"ECE {t:.2f}: sigma = {sigma_for_ece(P, E, t):.4f}")
