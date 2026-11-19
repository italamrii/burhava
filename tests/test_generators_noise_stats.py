import itertools

import numpy as np
import pytest

from burhava.generators import GENERATORS, STRUCTURES, GenConfig, structure_stats
from burhava.model import Problem
from burhava.noise import (
    corrupt_edges,
    independent_version,
    marginal_priors,
    perturb_calibration,
    population_ece,
    sigma_for_ece,
    with_verifier,
)
from burhava.stats import ece, paired_bootstrap


@pytest.mark.parametrize("gen", list(GENERATORS))
def test_generators_valid_over_grid(gen):
    rng = np.random.default_rng(0)
    for s, d, f, c, co in itertools.product(
        STRUCTURES, [2, 4, 6, 8], [0.0, 0.3], ["independent", "shared"], ["uniform", "varied"]
    ):
        g = GENERATORS[gen](GenConfig(s, d, f, c, co), rng)
        assert 1 <= g.n <= 14
        costs = np.array([a.cost for a in g.atoms])
        assert np.isclose(costs.mean(), 1.0)
        assert (c == "shared") == bool(g.sources)
        st = structure_stats(g)
        assert 0 <= st["p_y"] <= 1 and st["n_relevant"] <= g.n


def test_generator_false_rate_is_marginal_false_prob():
    rng = np.random.default_rng(1)
    for corr in ["independent", "shared"]:
        ms = []
        for _ in range(300):
            g = GENERATORS["layered"](GenConfig("and", 4, 0.2, corr), rng)
            ms.append(1 - marginal_priors(g))
        assert np.concatenate(ms).mean() == pytest.approx(0.2, abs=0.01)


def test_generators_deterministic():
    a = GENERATORS["rules"](GenConfig("mixed", 6), np.random.default_rng(5))
    b = GENERATORS["rules"](GenConfig("mixed", 6), np.random.default_rng(5))
    assert np.array_equal(Problem(a).joint, Problem(b).joint)


def test_independent_version_keeps_marginals_drops_correlation():
    g = GENERATORS["layered"](GenConfig("and", 4, 0.3, "shared"), np.random.default_rng(2))
    gi = independent_version(g)
    assert not gi.sources
    np.testing.assert_allclose(marginal_priors(gi), marginal_priors(g))
    np.testing.assert_allclose(
        Problem(gi).prior_belief().prob_atoms(), Problem(g).prior_belief().prob_atoms()
    )


def test_calibration_noise_hits_target_ece():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.5, 0.999, 20000)
    eps = rng.standard_normal(p.size)
    for t in (0.05, 0.1, 0.2):
        s = sigma_for_ece(p, eps, t)
        q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + s * eps)))
        assert population_ece(p, np.clip(q, 0.001, 0.999)) == pytest.approx(t, abs=1e-3)


def test_perturb_and_corrupt_keep_atoms_aligned():
    rng = np.random.default_rng(4)
    g = GENERATORS["rules"](GenConfig("mixed", 6, 0.2, "shared"), rng)
    g2 = perturb_calibration(g, 1.0, rng.standard_normal(g.n))
    g3, n_e, n_c = corrupt_edges(g2, 0.5, rng)
    g4 = with_verifier(g3, 0.05)
    assert [a.name for a in g4.atoms] == [a.name for a in g.atoms]
    assert all(a.alpha == 0.05 and a.beta == 0.05 for a in g4.atoms)
    assert 0 < n_c <= n_e
    Problem(g4)  # still a valid graph
    same, _, n0 = corrupt_edges(g, 0.0, rng)
    assert same is g and n0 == 0


def test_bootstrap_and_ece():
    rng = np.random.default_rng(0)
    r = paired_bootstrap(rng.normal(0.5, 1, 4000), n_boot=2000)
    assert r["lo"] < 0.5 < r["hi"] and r["lo"] > 0.4
    z = paired_bootstrap(np.zeros(100), n_boot=100)
    assert z["lo"] == z["hi"] == 0
    p = rng.random(50000)
    assert ece(p, rng.random(50000) < p) < 0.02
    assert ece(np.full(1000, 0.9), np.zeros(1000)) == pytest.approx(0.9)
