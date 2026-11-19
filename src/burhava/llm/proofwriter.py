"""ProofWriter loader, gold proof graphs, controlled corruption (Stage C, offline).

Data: ProofWriter V2020.12.3 (Tafjord, Dalvi & Clark 2021), CWA split, downloaded
by ``scripts/download_data.sh``. The archive's README does not state a licence;
check AI2's dataset page before redistributing anything derived from it.

Scope choices (see DECISIONS.md D21):
* only theories whose facts and rules contain no negation (725 of 985 theories
  in CWA depth-5 test), so truth under the closed-world assumption is plain
  forward chaining;
* only questions answered True with strategy ``proof`` and depth >= 2.

The Stage C scenario: the reasoner sees a *context* in which a controlled
fraction of facts has been corrupted (replaced by its negation). A verifier
with access to the clean theory can check individual facts. Atoms are the
facts and rules used by the gold proofs (rules become step atoms); derived
nodes are the proofs' intermediate conclusions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..graph import AssumptionGraph, Derivation, Literal

VARIABLES = {"someone", "something"}
Triple = tuple[str, str, str]  # (subject, relation, object), all positive


def parse_triple(rep: str) -> tuple[Triple, str]:
    """``("cat" "is" "nice" "+")`` -> ``(("cat", "is", "nice"), "+")``."""
    toks = re.findall(r'"([^"]*)"', rep)
    if len(toks) != 4:
        raise ValueError(f"bad triple {rep!r}")
    return (toks[0], toks[1], toks[2]), toks[3]


def parse_rule(rep: str) -> tuple[list[Triple], Triple]:
    """``(((a) (b)) -> (h))`` -> ``([a, b], h)`` (positive literals only)."""
    body_s, head_s = rep.split("->")
    body = [parse_triple(m)[0] for m in re.findall(r'\("[^()]*"\)', body_s)]
    head = parse_triple(re.findall(r'\("[^()]*"\)', head_s)[0])[0]
    return body, head


@dataclass
class Theory:
    """One ProofWriter theory (facts + rules), positive literals only."""

    id: str
    facts: dict[str, Triple]
    fact_text: dict[str, str]
    rules: dict[str, tuple[list[Triple], Triple]]
    rule_text: dict[str, str]
    questions: dict[str, dict] = field(default_factory=dict)


def load_theories(path: str | Path, limit: int | None = None) -> list[Theory]:
    """Negation-free theories from a ProofWriter ``meta-*.jsonl`` file."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            reps = [
                x["representation"] for x in list(r["triples"].values()) + list(r["rules"].values())
            ]
            if any('"~"' in s or '"-"' in s for s in reps):
                continue
            facts = {k: parse_triple(v["representation"])[0] for k, v in r["triples"].items()}
            rules = {k: parse_rule(v["representation"]) for k, v in r["rules"].items()}
            out.append(
                Theory(
                    r["id"],
                    facts,
                    {k: v["text"] for k, v in r["triples"].items()},
                    rules,
                    {k: v["text"] for k, v in r["rules"].items()},
                    r["questions"],
                )
            )
            if limit and len(out) >= limit:
                break
    return out


def _match(pattern: Triple, fact: Triple, binding: dict) -> dict | None:
    b = dict(binding)
    for p, x in zip(pattern, fact, strict=True):
        if p in VARIABLES:
            if p in b and b[p] != x:
                return None
            b[p] = x
        elif p != x:
            return None
    return b


def forward_chain(facts: set[Triple], rules: list[tuple[list[Triple], Triple]]) -> set[Triple]:
    """Closure of ``facts`` under the rules (variables: someone/something)."""
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for body, head in rules:
            bindings = [{}]
            for pat in body:
                bindings = [
                    nb for b in bindings for f in known if (nb := _match(pat, f, b)) is not None
                ]
                if not bindings:
                    break
            for b in bindings:
                new = tuple(b.get(t, t) for t in head)
                if new not in known:
                    known.add(new)
                    changed = True
    return known


# ----------------------------------------------------------- proof trees
def parse_proof(rep: str):
    """Parse a ``proofsWithIntermediates`` string, e.g.
    ``((triple1 triple2) -> (rule3 % int1))``.

    Returns a fact id (str) or ``("rule", rule_id, int_id, [premise trees])``.
    """
    toks = re.findall(r"\(|\)|->|%|[A-Za-z0-9_]+", rep)
    pos = 0

    def nest():  # tokens -> nested python lists
        nonlocal pos
        out = []
        while pos < len(toks):
            t = toks[pos]
            pos += 1
            if t == "(":
                out.append(nest())
            elif t == ")":
                return out
            else:
                out.append(t)
        return out

    def interp(x):
        if isinstance(x, str):
            return x
        if "->" in x:
            k = x.index("->")
            prem = []
            for c in x[:k]:
                v = interp(c)
                prem.extend(v if isinstance(v, list) else [v])
            rhs = x[k + 1]
            while isinstance(rhs, list) and len(rhs) == 1:
                rhs = rhs[0]
            rule, iid = (rhs[0], rhs[2]) if isinstance(rhs, list) else (rhs, None)
            return ("rule", rule, iid, prem)
        if len(x) == 1:
            return interp(x[0])
        return [interp(c) for c in x]

    return interp(nest())


@dataclass
class Instance:
    """A Stage C item: corrupted context, gold graph, and the true answer."""

    theory_id: str
    question: str
    context: list[str]  # sentences the reasoner sees (some corrupted)
    graph: AssumptionGraph  # gold dependency graph (priors filled in later)
    atom_text: list[str]  # text of each atom (fact sentence or rule sentence)
    atom_ids: list[str]  # ProofWriter ids (triple*/rule*)
    corrupted: list[str]  # ids of corrupted facts
    y_true: bool
    gold_edges: set[tuple[str, str]]  # (premise id, conclusion id)


def negate_text(sentence: str) -> str:
    """'The cat is nice.' -> 'The cat is not nice.'; 'The cat likes the dog.' ->
    'The cat does not like the dog.'"""
    s = sentence.rstrip(".")
    m = re.match(r"^(.*?) is (.*)$", s)
    if m:
        return f"{m.group(1)} is not {m.group(2)}."
    m = re.match(r"^(\S+ \S+) (\w+?)s (.*)$", s)
    if m:
        return f"{m.group(1)} does not {m.group(2)} {m.group(3)}."
    return f"It is not true that {s[0].lower() + s[1:]}."


def build_instance(
    th: Theory, qid: str, corrupt_rate: float, rng: np.random.Generator
) -> Instance | None:
    """Gold graph for question ``qid``; corrupt each theory fact with prob.
    ``corrupt_rate``; recompute the truth by forward chaining."""
    q = th.questions[qid]
    if q["strategy"] != "proof" or not q["answer"] or q["QDep"] < 2:
        return None
    if "NAF" in q["proofs"]:
        return None
    goal = parse_triple(q["representation"])[0]
    g = AssumptionGraph()
    atom_ids: list[str] = []
    gold_edges: set[tuple[str, str]] = set()
    node_of: dict[str, str] = {}  # intermediate id/text -> node name
    proofs = [parse_proof(p["representation"]) for p in q["proofsWithIntermediates"]]

    def atom(pid: str, kind: str) -> str:
        if pid not in node_of:
            g.add_atom(pid, 0.5, kind=kind)  # priors are set later by the pipeline
            atom_ids.append(pid)
            node_of[pid] = pid
        return pid

    derivs: dict[str, list[Derivation]] = {}
    producer: dict[str, str] = {}  # derived node -> rule that produced it
    order: list[str] = []

    def walk(t) -> str:
        if isinstance(t, str):
            return atom(t, "assumption")
        _, rule, iid, prem = t
        names = [walk(c) for c in prem]
        step = atom(rule, "step")
        key = f"{rule}:{'+'.join(sorted(names))}"
        d = Derivation(tuple(Literal(n) for n in names), step)
        node = f"D_{key}"
        if node not in derivs:
            derivs[node] = [d]
            order.append(node)
        for n in names:
            gold_edges.add((producer.get(n, n), rule))
        producer[node] = rule
        return node

    tops = [walk(p) for p in proofs]
    for node in order:
        g.add_derived(node, derivs[node])
    g.add_derived("Y", [Derivation((Literal(t),)) for t in dict.fromkeys(tops)])
    g.set_conclusion("Y")
    if g.n > 14:
        return None
    # corruption of facts (anywhere in the theory, used or distractor)
    corrupted = [k for k in th.facts if rng.random() < corrupt_rate]
    clean_facts = {v for k, v in th.facts.items() if k not in corrupted}
    y_true = goal in forward_chain(clean_facts, list(th.rules.values()))
    context = [
        negate_text(th.fact_text[k]) if k in corrupted else th.fact_text[k] for k in th.facts
    ] + list(th.rule_text.values())
    texts = [th.fact_text.get(i, th.rule_text.get(i, "")) for i in atom_ids]
    return Instance(
        th.id, q["question"], context, g, texts, atom_ids, corrupted, bool(y_true), gold_edges
    )


def edge_f1(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict:
    """Precision, recall and F1 of recovered dependency edges."""
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}
