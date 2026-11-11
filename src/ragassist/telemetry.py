"""Lightweight per-request tracing: spans, tokens, latency, cost.

Exports to Cloud Trace / Langfuse later via OpenTelemetry; this in-process
recorder is what the eval harness and /metrics use.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from ragassist.config import PRICING


@dataclass
class Span:
    name: str
    start: float
    end: float = 0.0
    attrs: dict = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return (self.end - self.start) * 1000


@dataclass
class RequestTrace:
    spans: list[Span] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    first_token_at: float | None = None
    started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def span(self, name: str, **attrs):
        s = Span(name, time.perf_counter(), attrs=attrs)
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            self.spans.append(s)

    def record_llm(self, model: str, tokens_in: int, tokens_out: int):
        pin, pout = PRICING.get(model, (0.0, 0.0))
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += (tokens_in * pin + tokens_out * pout) / 1_000_000

    def summary(self) -> dict:
        total = time.perf_counter() - self.started
        return {
            "latency_ms": round(total * 1000, 1),
            "ttft_ms": round((self.first_token_at - self.started) * 1000, 1) if self.first_token_at else None,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_per_sec": round(self.tokens_out / total, 1) if total > 0 else None,
            "cost_usd": round(self.cost_usd, 6),
            "spans": {s.name: round(s.ms, 1) for s in self.spans},
        }
