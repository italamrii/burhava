"""Burhava: decision-risk-aware verification selection for reasoning chains.

The selection rule is classical Value of Information (VoI); this package tests it
empirically. Core modules: ``graph`` (AND/OR assumption graphs), ``model``
(joint P(Z), verifier noise, Bayes update), ``risk`` (rho, R(S), Delta).
"""

from .graph import AssumptionGraph, Derivation, Literal, neg
from .model import Belief, Problem
from .risk import (
    ABSTAIN,
    ACCEPT,
    REJECT,
    Loss,
    bayes_decision,
    bayes_risk,
    marginal_value,
    set_risk,
)

__all__ = [
    "ABSTAIN",
    "ACCEPT",
    "REJECT",
    "AssumptionGraph",
    "Belief",
    "Derivation",
    "Literal",
    "Loss",
    "Problem",
    "bayes_decision",
    "bayes_risk",
    "marginal_value",
    "neg",
    "set_risk",
]
