"""AND/OR assumption graphs.

Plain-language picture
----------------------
An LLM reasoning chain rests on *atoms*: binary facts that are either true (1) or
false (0). There are two kinds of atom:

* **assumptions** ``Z_i`` - premises the chain takes for granted;
* **steps** - "this inference step is valid" variables.

Both kinds are random, both can (optionally) be verified. In the maths we call
every atom ``Z_i`` and write ``Z = (Z_1, ..., Z_n)`` for all of them together.

*Derived nodes* are built from atoms by rules:

* a **derivation** holds if all its premises hold AND its step (if any) is valid;
* a **derived node** is true if ANY of its alternative derivations holds.

So a derivation is an AND, a derived node is an OR of ANDs. The *conclusion*
``Y = f(Z)`` is one chosen node. A premise may be negated (``neg("Z2")``), which
lets us express XOR and closed-world negation as in ProofWriter.

Atoms may cite a *source* ``k``. Atoms that cite the same source are
correlated through a hidden reliability variable ``U_k`` (see ``model.py``).

Nodes are referenced by name. A node must be defined before it is used, which
guarantees the graph has no cycles and that insertion order is a valid
topological order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Hard cap on the number of atoms. Exact enumeration touches 2^n states, so
#: n = 20 means ~1M states (about 30 MB of working memory). See DECISIONS.md.
MAX_ATOMS = 20

ASSUMPTION = "assumption"
STEP = "step"


@dataclass(frozen=True)
class Source:
    """A latent source ``k`` with reliability variable ``U_k``.

    Args:
        name: identifier.
        u_probs: ``P(U_k = u)`` for each latent value ``u = 0, 1, ...``. Sums to 1.
    """

    name: str
    u_probs: tuple[float, ...]


@dataclass(frozen=True)
class Atom:
    """A base binary variable ``Z_i`` (an assumption or a step-validity variable).

    Args:
        name: identifier.
        kind: ``"assumption"`` or ``"step"``.
        prior: ``P(Z_i = 1 | U_{s(i)} = u)`` for each latent value ``u`` of the
            cited source. Length 1 (a plain prior ``P(Z_i = 1)``) when the atom
            cites no source.
        source: name of the cited source ``s(i)``, or ``None`` (independent atom).
        cost: verification cost ``c_i``.
        alpha: verifier false-positive rate ``alpha_i = P(O_i = 1 | Z_i = 0)``.
        beta: verifier false-negative rate ``beta_i = P(O_i = 0 | Z_i = 1)``.
        verifiable: whether a policy may check this atom at all.
    """

    name: str
    kind: str
    prior: tuple[float, ...]
    source: str | None
    cost: float
    alpha: float
    beta: float
    verifiable: bool


@dataclass(frozen=True)
class Literal:
    """A reference to a node, possibly negated (``NOT node``)."""

    node: str
    negated: bool = False


def neg(name: str) -> Literal:
    """Return the negated literal ``NOT name``."""
    return Literal(name, negated=True)


@dataclass(frozen=True)
class Derivation:
    """One way to derive a node: ``AND(premises) AND step``.

    Args:
        premises: literals that must all hold.
        step: name of a step-validity atom that must hold, or ``None`` if the
            step is treated as certainly valid.
    """

    premises: tuple[Literal, ...]
    step: str | None = None


@dataclass(frozen=True)
class DerivedNode:
    """A node that is true iff ANY of its derivations holds (an OR of ANDs)."""

    name: str
    derivations: tuple[Derivation, ...]


def _as_literal(x: str | Literal) -> Literal:
    return x if isinstance(x, Literal) else Literal(x)


def _check_prob(x: float, what: str) -> None:
    if not (0.0 <= x <= 1.0) or not np.isfinite(x):
        raise ValueError(f"{what} must be in [0, 1], got {x}")


@dataclass
class AssumptionGraph:
    """Builder and evaluator for an AND/OR assumption graph.

    Atom ``i`` is the ``i``-th atom added (0-based). The binary state of all
    atoms is encoded as an integer whose bit ``i`` equals ``Z_i``.
    """

    sources: dict[str, Source] = field(default_factory=dict)
    atoms: list[Atom] = field(default_factory=list)
    derived: list[DerivedNode] = field(default_factory=list)
    conclusion: str | None = None
    _index: dict[str, tuple[str, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    def add_source(self, name: str, u_probs: tuple[float, ...] | list[float]) -> str:
        """Add a latent source ``U_k`` with distribution ``P(U_k = u) = u_probs[u]``."""
        if name in self.sources:
            raise ValueError(f"duplicate source {name!r}")
        probs = tuple(float(p) for p in u_probs)
        if len(probs) < 1:
            raise ValueError("source needs at least one latent value")
        for p in probs:
            _check_prob(p, "u_probs entry")
        if abs(sum(probs) - 1.0) > 1e-9:
            raise ValueError(f"u_probs must sum to 1, got {sum(probs)}")
        self.sources[name] = Source(name, probs)
        return name

    def add_atom(
        self,
        name: str,
        prior: float | tuple[float, ...] | list[float],
        *,
        kind: str = ASSUMPTION,
        source: str | None = None,
        cost: float = 1.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        verifiable: bool = True,
    ) -> str:
        """Add a base atom ``Z_i``. See :class:`Atom` for the meaning of each argument.

        ``prior`` is a float for an independent atom, or one probability per
        latent value of ``source`` for a source-correlated atom.
        """
        self._check_new_name(name)
        if kind not in (ASSUMPTION, STEP):
            raise ValueError(f"kind must be {ASSUMPTION!r} or {STEP!r}")
        if len(self.atoms) >= MAX_ATOMS:
            raise ValueError(f"at most {MAX_ATOMS} atoms supported (exact enumeration)")
        pr = (float(prior),) if np.isscalar(prior) else tuple(float(p) for p in prior)
        for p in pr:
            _check_prob(p, f"prior of {name}")
        if source is None:
            if len(pr) != 1:
                raise ValueError("an atom without a source takes a single prior")
        else:
            if source not in self.sources:
                raise ValueError(f"unknown source {source!r}")
            if len(pr) != len(self.sources[source].u_probs):
                raise ValueError(f"prior of {name} needs one value per latent value of {source!r}")
        _check_prob(alpha, "alpha")
        _check_prob(beta, "beta")
        if not cost > 0:
            raise ValueError("cost must be positive")
        self._index[name] = ("atom", len(self.atoms))
        self.atoms.append(
            Atom(name, kind, pr, source, float(cost), float(alpha), float(beta), verifiable)
        )
        return name

    def add_step(self, name: str, prior: float | tuple[float, ...], **kw) -> str:
        """Add a step-validity atom (shorthand for ``add_atom(kind="step")``)."""
        return self.add_atom(name, prior, kind=STEP, **kw)

    def add_derived(self, name: str, derivations: list[Derivation] | list[tuple]) -> str:
        """Add a derived node: true iff ANY derivation holds.

        Each derivation is a :class:`Derivation` or a tuple
        ``(premises,)`` / ``(premises, step_name)``.
        """
        self._check_new_name(name)
        ders: list[Derivation] = []
        for d in derivations:
            if not isinstance(d, Derivation):
                premises = tuple(_as_literal(p) for p in d[0])
                step = d[1] if len(d) > 1 else None
                d = Derivation(premises, step)
            if not d.premises and d.step is None:
                raise ValueError("a derivation needs at least one premise or a step")
            for lit in d.premises:
                if lit.node not in self._index:
                    raise ValueError(f"premise {lit.node!r} used before definition")
            if d.step is not None:
                kind_idx = self._index.get(d.step)
                if kind_idx is None or kind_idx[0] != "atom":
                    raise ValueError(f"step {d.step!r} must be a defined atom")
            ders.append(d)
        if not ders:
            raise ValueError("a derived node needs at least one derivation")
        self._index[name] = ("derived", len(self.derived))
        self.derived.append(DerivedNode(name, tuple(ders)))
        return name

    def add_and(self, name: str, premises: list[str | Literal], step: str | None = None) -> str:
        """Add ``name = AND(premises) [AND step]`` (one derivation)."""
        return self.add_derived(name, [Derivation(tuple(map(_as_literal, premises)), step)])

    def add_or(self, name: str, premises: list[str | Literal]) -> str:
        """Add ``name = OR(premises)`` (one single-premise derivation per premise)."""
        return self.add_derived(name, [Derivation((_as_literal(p),)) for p in premises])

    def set_conclusion(self, name: str) -> None:
        """Choose the node whose truth value is the conclusion ``Y``."""
        if name not in self._index:
            raise ValueError(f"unknown node {name!r}")
        self.conclusion = name

    def _check_new_name(self, name: str) -> None:
        if name in self._index:
            raise ValueError(f"duplicate node name {name!r}")

    # ------------------------------------------------------------- queries
    @property
    def n(self) -> int:
        """Number of atoms ``n``."""
        return len(self.atoms)

    def atom_index(self, name: str) -> int:
        """Index ``i`` of the atom called ``name``."""
        kind, idx = self._index[name]
        if kind != "atom":
            raise KeyError(f"{name!r} is a derived node, not an atom")
        return idx

    def evaluate(self, bits: np.ndarray) -> dict[str, np.ndarray]:
        """Truth value of every node for a batch of atom states.

        Args:
            bits: boolean array of shape ``(n, N)``; ``bits[i, s]`` is ``Z_i`` in
                state ``s``.

        Returns:
            ``{node_name: bool array of shape (N,)}`` for atoms and derived nodes.
        """
        if bits.shape[0] != self.n:
            raise ValueError("bits must have one row per atom")
        vals: dict[str, np.ndarray] = {a.name: bits[i] for i, a in enumerate(self.atoms)}
        N = bits.shape[1]
        for node in self.derived:
            out = np.zeros(N, dtype=bool)
            for d in node.derivations:
                term = np.ones(N, dtype=bool)
                for lit in d.premises:
                    v = vals[lit.node]
                    term &= ~v if lit.negated else v
                if d.step is not None:
                    term &= vals[d.step]
                out |= term
            vals[node.name] = out
        return vals

    def conclusion_values(self, bits: np.ndarray) -> np.ndarray:
        """``Y = f(Z)`` for each column of ``bits`` (shape ``(n, N)``)."""
        if self.conclusion is None:
            raise ValueError("conclusion not set")
        return self.evaluate(bits)[self.conclusion]

    def degree(self) -> np.ndarray:
        """Out-degree of each atom: how many derivations use it (premise or step)."""
        deg = np.zeros(self.n, dtype=int)
        for node in self.derived:
            for d in node.derivations:
                for lit in d.premises:
                    k, i = self._index[lit.node]
                    if k == "atom":
                        deg[i] += 1
                if d.step is not None:
                    deg[self._index[d.step][1]] += 1
        return deg
