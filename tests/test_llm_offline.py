"""Stage C code paths, offline (no data download, no API)."""

import json

import numpy as np
import pytest

from burhava.llm import prompts
from burhava.llm.client import CallLog, GroqClient, MockLLM
from burhava.llm.proofwriter import (
    build_instance,
    edge_f1,
    forward_chain,
    load_theories,
    negate_text,
    parse_proof,
)
from burhava.model import Problem

THEORY = {
    "id": "T1",
    "maxD": 2,
    "NFact": 2,
    "NRule": 2,
    "theory": "...",
    "triples": {
        "triple1": {"text": "The cat is nice.", "representation": '("cat" "is" "nice" "+")'},
        "triple2": {
            "text": "The cat likes the dog.",
            "representation": '("cat" "likes" "dog" "+")',
        },
    },
    "rules": {
        "rule1": {
            "text": "If someone is nice and they like the dog then they are kind.",
            "representation": '((("someone" "is" "nice" "+") ("someone" "likes" "dog" "+"))'
            ' -> ("someone" "is" "kind" "+"))',
        },
        "rule2": {
            "text": "If someone is kind then they are big.",
            "representation": '((("someone" "is" "kind" "+")) -> ("someone" "is" "big" "+"))',
        },
    },
    "questions": {
        "Q1": {
            "question": "The cat is big.",
            "answer": True,
            "QDep": 2,
            "strategy": "proof",
            "proofs": "[(((triple1 triple2) -> rule1) -> rule2)]",
            "representation": '("cat" "is" "big" "+")',
            "proofsWithIntermediates": [
                {
                    "representation": "((((triple1 triple2) -> (rule1 % int2))) -> (rule2 % int1))",
                    "intermediates": {},
                }
            ],
        },
    },
}


@pytest.fixture
def theory(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(THEORY) + "\n")
    return load_theories(p)[0]


def test_parse_and_forward_chain(theory):
    assert parse_proof("((triple1 triple2) -> (rule1 % int2))") == (
        "rule",
        "rule1",
        "int2",
        ["triple1", "triple2"],
    )
    closure = forward_chain(set(theory.facts.values()), list(theory.rules.values()))
    assert ("cat", "is", "big") in closure


def test_gold_graph_and_corruption(theory):
    inst = build_instance(theory, "Q1", 0.0, np.random.default_rng(0))
    assert inst.y_true and inst.atom_ids == ["triple1", "triple2", "rule1", "rule2"]
    assert inst.gold_edges == {("triple1", "rule1"), ("triple2", "rule1"), ("rule1", "rule2")}
    p = Problem(inst.graph)
    assert p.y[-1] and not p.y[0]  # all true -> Y; all false -> not Y
    bad = build_instance(theory, "Q1", 1.0, np.random.default_rng(0))
    assert not bad.y_true and "The cat is not nice." in bad.context


def test_negate_text_and_f1():
    assert negate_text("The cat is nice.") == "The cat is not nice."
    assert negate_text("The cat likes the dog.") == "The cat does not like the dog."
    r = edge_f1({("a", "b"), ("c", "d")}, {("a", "b")})
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_prompts_and_mock_logging(theory):
    inst = build_instance(theory, "Q1", 0.0, np.random.default_rng(0))
    log = CallLog()
    llm = MockLLM(
        lambda p: "STEP 1: USES [0, 1] -> kind\nSTEP 2: USES [S1] -> big\nANSWER: TRUE", log
    )
    out = llm.complete(prompts.chain_prompt(inst.context, inst.question), tag="a10")
    assert prompts.parse_chain(out) == [(1, ["0", "1"]), (2, ["S1"])]
    assert log.calls["a10"] == 1
    assert prompts.parse_number("I'd say 85.") == 85 and prompts.parse_yes("YES.") == 1


def test_real_client_refuses_without_approval(monkeypatch):
    monkeypatch.delenv("BURHAVA_ALLOW_SPEND", raising=False)
    with pytest.raises(PermissionError):
        GroqClient("any-model").complete("hi", tag="x")
