import numpy as np
import pytest

from burhava.graph import MAX_ATOMS, AssumptionGraph, neg


def _all_bits(n):
    s = np.arange(1 << n)
    return ((s[None, :] >> np.arange(n)[:, None]) & 1).astype(bool)


def test_and_or_neg_step_semantics():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    g.add_atom("b", 0.5)
    g.add_step("s", 0.9)
    g.add_derived("d", [(["a", neg("b")], "s"), (["b"],)])  # (a & !b & s) | b
    g.set_conclusion("d")
    bits = _all_bits(3)
    a, b, s = bits
    np.testing.assert_array_equal(g.conclusion_values(bits), (a & ~b & s) | b)


def test_validation_errors():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    with pytest.raises(ValueError):
        g.add_atom("a", 0.5)  # duplicate
    with pytest.raises(ValueError):
        g.add_and("x", ["missing"])  # used before definition
    with pytest.raises(ValueError):
        g.add_atom("p", 1.5)  # not a probability
    with pytest.raises(ValueError):
        g.add_atom("q", (0.1, 0.2))  # sourced prior without a source
    g.add_source("s", [0.3, 0.7])
    with pytest.raises(ValueError):
        g.add_atom("r", (0.1, 0.2, 0.3), source="s")  # wrong length
    with pytest.raises(ValueError):
        g.add_source("t", [0.3, 0.3])  # does not sum to 1
    g.add_and("x", ["a"])
    with pytest.raises(ValueError):
        g.add_derived("y", [(["a"], "x")])  # step must be an atom


def test_atom_cap():
    g = AssumptionGraph()
    for i in range(MAX_ATOMS):
        g.add_atom(f"z{i}", 0.5)
    with pytest.raises(ValueError):
        g.add_atom("one_more", 0.5)


def test_degree():
    g = AssumptionGraph()
    g.add_atom("a", 0.5)
    g.add_atom("b", 0.5)
    g.add_step("s", 0.5)
    g.add_and("x", ["a", "b"], step="s")
    g.add_or("y", ["a", "x"])
    np.testing.assert_array_equal(g.degree(), [2, 1, 1])
