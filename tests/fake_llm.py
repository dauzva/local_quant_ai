"""Offline stand-in for OpenRouterClient: answers explore/evolve prompts with
snippet blocks built from classic strategy templates + random parameters.
Used by the tests and for calibrating acceptance thresholds without API calls."""
from __future__ import annotations

import json
import random
import re

from src.llm import LLMTextResponse, UsageTracker

from src.seeds import TEMPLATES  # noqa: E402


def _block(rng: random.Random, i: int, parent: str | None = None) -> str:
    fam, idea, ranges, body = rng.choice(TEMPLATES)
    params = {}
    for k, (lo, hi) in ranges.items():
        params[k] = rng.randint(lo, hi) if isinstance(lo, int) and isinstance(hi, int) else round(rng.uniform(lo, hi), 3)
    name = re.sub(r"\W+", "_", idea) + f"_{rng.randint(1000, 9999)}"
    lines = ["===S===", f"name: {name}", f"family: {fam}", f"idea: {idea}",
             f"sl: {rng.choice([0.015, 0.02, 0.03, 0.05])}", f"tp: {rng.choice([0.03, 0.06, 0.1, 0.2])}",
             f"params: {json.dumps(params)}"]
    if parent:
        lines.append(f"parent: {parent}")
    return "\n".join(lines) + "\n```python\n" + body + "\n```"


class FakeClient:
    """Drop-in for OpenRouterClient (complete_text + usage + total_cost)."""

    def __init__(self, cfg=None, junk_every: int = 7):
        self.usage = UsageTracker()
        self.calls = 0
        self.junk_every = junk_every
        self.rng = random.Random(7)

    def total_cost(self) -> float:
        return 0.0

    def complete_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> LLMTextResponse:
        self.calls += 1
        n = int(re.search(r"Write exactly (\d+) blocks", prompt).group(1))
        parents = re.findall(r"### PARENT (\w+)", prompt)
        blocks = []
        for i in range(n):
            parent = parents[i // max(1, n // max(1, len(parents)))] if parents and i // max(1, n // len(parents)) < len(parents) else None
            b = _block(self.rng, i, parent)
            if self.junk_every and (i + 1) % self.junk_every == 0:  # sprinkle in invalid strategies
                b = b.replace("```python\n", "```python\nz = getattr(close, 'shift')(1)\n")
            blocks.append(b)
        self.usage.record("fake/model:free", len(prompt) // 4, sum(len(b) for b in blocks) // 4, 0.0, False, 0.01)
        return LLMTextResponse(text="\n\n".join(blocks), model_used="fake/model:free")
