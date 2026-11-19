"""LLM access with per-arm call accounting. No third-party dependency.

``GroqClient`` talks to an OpenAI-compatible chat endpoint with the standard
library. It refuses to make any request unless the environment variable
``BURHAVA_ALLOW_SPEND=1`` is set: API budget must be approved first (rule 5).
``MockLLM`` returns scripted answers for offline tests and dry runs.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CallLog:
    """Counts calls and characters per tag (e.g. per arm), for fair compute."""

    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    chars_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    chars_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, tag: str, prompt: str, answer: str) -> None:
        self.calls[tag] += 1
        self.chars_in[tag] += len(prompt)
        self.chars_out[tag] += len(answer)


class GroqClient:
    """Minimal chat-completions client (Groq's OpenAI-compatible API)."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str, log: CallLog | None = None, timeout: float = 60):
        self.model, self.log, self.timeout = model, log or CallLog(), timeout

    def complete(
        self, prompt: str, *, tag: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        if os.environ.get("BURHAVA_ALLOW_SPEND") != "1":
            raise PermissionError("API spending not approved: set BURHAVA_ALLOW_SPEND=1")
        body = json.dumps(
            {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            self.URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', 'proxy-injected')}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())["choices"][0]["message"]["content"]
        self.log.add(tag, prompt, out)
        return out


class MockLLM:
    """Offline stand-in: ``fn(prompt) -> answer``; logs like the real client."""

    def __init__(self, fn: Callable[[str], str], log: CallLog | None = None):
        self.fn, self.log = fn, log or CallLog()

    def complete(
        self, prompt: str, *, tag: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        out = self.fn(prompt)
        self.log.add(tag, prompt, out)
        return out
