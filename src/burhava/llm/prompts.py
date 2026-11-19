"""Stage C prompt templates and parsers (draft; frozen before any paid run)."""

from __future__ import annotations

import re


def numbered_context(context: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(context))


def chain_prompt(context: list[str], question: str) -> str:
    """Ask for a step-by-step proof that cites sentence numbers (graph recovery)."""
    return (
        "Below are numbered statements.\n" + numbered_context(context) + "\n\n"
        f"Claim: {question}\n"
        "Prove or refute the claim step by step. Write each step on its own line as\n"
        "STEP k: USES [i, j, ...] -> <new conclusion>\n"
        "where the numbers are statement numbers or earlier steps written as Sk.\n"
        "End with a line ANSWER: TRUE or ANSWER: FALSE."
    )


STEP_RE = re.compile(r"STEP\s*(\d+)\s*:\s*USES\s*\[([^\]]*)\]", re.I)


def parse_chain(text: str) -> list[tuple[int, list[str]]]:
    """``[(step number, [cited ids like '3' or 'S1'])]``."""
    out = []
    for m in STEP_RE.finditer(text):
        cites = [c.strip() for c in m.group(2).split(",") if c.strip()]
        out.append((int(m.group(1)), cites))
    return out


def confidence_prompt(context: list[str], i: int) -> str:
    """Verbalised confidence that statement i is correct (0-100)."""
    return (
        "Statements:\n" + numbered_context(context) + "\n\n"
        f"How confident are you that statement [{i}] is correct? "
        "Reply with a single number from 0 to 100."
    )


def verifier_prompt(reference: list[str], claim: str) -> str:
    """The verifier sees the clean reference statements and one claim."""
    return (
        "Reference statements:\n" + "\n".join(reference) + "\n\n"
        f"Claim: {claim}\nIs the claim stated or directly supported by the reference? "
        "Reply YES or NO."
    )


def parse_number(text: str, default: float = 50.0) -> float:
    m = re.search(r"\d+(\.\d+)?", text)
    return min(max(float(m.group()), 0.0), 100.0) if m else default


def parse_yes(text: str) -> int:
    return int(bool(re.search(r"\byes\b", text, re.I)))
