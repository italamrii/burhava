"""M1 'done' criterion: an n = 20 risk evaluation takes < 1 s (slow test)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


@pytest.mark.slow
def test_n20_risk_under_one_second():
    from bench_m1 import build_graph

    from burhava import Loss, Problem, set_risk

    b = Problem(build_graph(np.random.default_rng(0))).prior_belief()
    L = Loss.from_costs(10, 1, 0.6)
    t0 = time.perf_counter()
    set_risk(b, L, [0, 1, 2, 3, 4])
    assert time.perf_counter() - t0 < 1.0
