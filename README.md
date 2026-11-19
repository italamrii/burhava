# Burhava

**Which step of a reasoning chain should you verify?** Burhava studies verification selection for LLM reasoning chains. Each candidate check is scored by how much it is expected to reduce the *loss of the final decision* (accept / reject / abstain), divided by its cost. This is classical **Value of Information**. We do not claim the rule is new. The question is empirical: does it beat confidence-based selection, and does the advantage survive realistic estimation noise?

## Status
| Milestone | State |
|---|---|
| M1 exact engine | ✅ done: acceptance tests pass, n=20 R(S) in ≤ 21 ms |
| M2 policies + simulator | not started |
| M3 Stage A (synthetic, pre-registered) | not started |
| M4 Stage B (noise robustness) | not started |
| M5 Stage C (real LLMs) | not started |
| M6 write-up | not started |

**No experimental results yet.** Results tables will appear here only with the command that produced them.

## Quickstart
```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
uv run pytest -q                      # all tests incl. acceptance tests
uv run python scripts/bench_m1.py     # n=20 timing
```

```python
from burhava import AssumptionGraph, Problem, Loss, set_risk, marginal_value

g = AssumptionGraph()
g.add_atom("Z1", 0.90)
g.add_atom("Z2", 0.60)
g.add_atom("Z3", 0.55)
g.add_or("Z2orZ3", ["Z2", "Z3"])
g.add_and("Y", ["Z1", "Z2orZ3"])
g.set_conclusion("Y")
b = Problem(g).prior_belief()
L = Loss.from_costs(false_accept=10, false_reject=1, abstain=0.6)
print(set_risk(b, L), [marginal_value(b, L, [], i) for i in range(3)])  # 0.6 [0.06, 0.042, 0.027]
```

## What this is not
* Not a new decision principle: the rule is VoI (Howard 1966; Krause & Guestrin 2009).
* Not scalable inference: the engine enumerates all 2^n states exactly (n ≤ 20).
* Not yet evidence of anything: see the status table.

## Docs
[Research brief](docs/RESEARCH_BRIEF.md) · [Math](docs/MATH.md) · [Decisions](docs/DECISIONS.md) · [Experiments](docs/EXPERIMENTS.md) · [Related work](docs/RELATED_WORK.md)
