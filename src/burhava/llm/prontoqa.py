"""PrOntoQA-OOD "ProofsOnly" loader (Saparov & He 2023; Apache-2.0).

Each item has a context (question), a query ("Prove: ...") and a gold linear
chain of thought. Context sentences used by the chain become atoms: the first
fact is an assumption, each rule ("Every X is a Y") a step atom; the chain is a
sequence of derived nodes. Corrupting a used context sentence breaks the only
proof, so the conclusion becomes unprovable (false under the closed world).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..graph import AssumptionGraph, Derivation, Literal
from .proofwriter import Instance, negate_text


def _sentences(text: str) -> list[str]:
    return [s.strip() + "." for s in text.split(".") if s.strip()]


def load_items(path: str | Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    return [v["test_example"] for v in d.values()]


def build_instance(
    item: dict, idx: int, corrupt_rate: float, rng: np.random.Generator
) -> Instance | None:
    ctx = _sentences(item["question"])
    cot = item["chain_of_thought"]
    used = [s for s in cot if s in ctx]
    if len(used) < 2 or len(used) > 14:
        return None
    g = AssumptionGraph()
    ids = {s: f"s{ctx.index(s)}" for s in used}
    for k, s in enumerate(used):
        g.add_atom(ids[s], 0.5, kind="assumption" if k == 0 else "step")
    prev, edges, n = ids[used[0]], set(), 0
    for s in used[1:]:
        node = f"D{n}"
        g.add_derived(node, [Derivation((Literal(prev),), ids[s])])
        edges.add((prev if not prev.startswith("D") else f"r{n - 1}", ids[s]))
        prev, n = node, n + 1
    g.set_conclusion(prev)
    corrupted = [f"s{i}" for i in range(len(ctx)) if rng.random() < corrupt_rate]
    y_true = not any(ids[s] in corrupted for s in used)
    context = [negate_text(s) if f"s{i}" in corrupted else s for i, s in enumerate(ctx)]
    q = re.sub(r"^Prove:\s*", "", item["query"])
    return Instance(
        f"prontoqa-{idx}", q, context, g, used, [ids[s] for s in used], corrupted, y_true, edges
    )
