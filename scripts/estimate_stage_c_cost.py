"""Stage C token/call estimate (no API calls). Usage:
python scripts/estimate_stage_c_cost.py --n-test 600 --n-cal 200 --seed 0

Builds the planned instances from the real data, renders every planned prompt,
and counts characters. Tokens are approximated as characters / 4 (a common rule
of thumb for English; the real tokenizer may differ by +-20%).

Planned calls per instance (n = number of atoms):
  chain (graph recovery + answer)         1   call,  ~400 output tokens
  verbalised confidence per atom          n   calls, ~4 output tokens
  sampling-consistency confidence (k=5)   5n  calls, ~4 output tokens (temperature 1)
  verifier per atom (cached, shared)      n   calls, ~3 output tokens
  self-consistency arm samples (max 5)    5   calls, ~400 output tokens
"""

import argparse
import glob

import numpy as np

from burhava.llm import prompts, prontoqa, proofwriter

PW = "data/raw/proofwriter/x/proofwriter-dataset-V2020.12.3/CWA/depth-5/meta-test.jsonl"
PQ = "data/raw/prontoqa/x/*ProofsOnly*.json"

ap = argparse.ArgumentParser()
ap.add_argument("--n-test", type=int, default=600)
ap.add_argument("--n-cal", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
N = a.n_test + a.n_cal
rng = np.random.default_rng(a.seed)

pw = []
for th in proofwriter.load_theories(PW):
    for qid in th.questions:
        inst = proofwriter.build_instance(th, qid, 0.1, rng)
        if inst:
            pw.append(inst)
pq = []
for f in sorted(glob.glob(PQ)):
    for k, item in enumerate(prontoqa.load_items(f)):
        inst = prontoqa.build_instance(item, k, 0.1, rng)
        if inst:
            pq.append(inst)
print(f"available: ProofWriter {len(pw)}, PrOntoQA {len(pq)}")

tot_in = tot_out = tot_calls = 0
for name, pool in [("ProofWriter", pw), ("PrOntoQA", pq)]:
    pick = [pool[i] for i in rng.choice(len(pool), size=min(N, len(pool)), replace=False)]
    cin = cout = calls = 0
    for inst in pick:
        n = inst.graph.n
        chain = len(prompts.chain_prompt(inst.context, inst.question))
        conf = len(prompts.confidence_prompt(inst.context, 0))
        ver = len(prompts.verifier_prompt(inst.context, inst.atom_text[0]))
        cin += 6 * chain + 6 * n * conf + n * ver
        cout += 4 * (6 * 400 + 6 * n * 4 + n * 3)
        calls += 6 + 7 * n
    print(
        f"{name}: {len(pick)} instances, {calls} calls, "
        f"~{cin / 4 / 1e6:.2f}M input tokens, ~{cout / 4 / 1e6:.2f}M output tokens"
    )
    tot_in += cin / 4
    tot_out += cout / 4
    tot_calls += calls
print(f"TOTAL: {tot_calls} calls, ~{tot_in / 1e6:.2f}M input, ~{tot_out / 1e6:.2f}M output tokens")
print("cost = input_M * price_in_per_M + output_M * price_out_per_M (prices: provider page)")
