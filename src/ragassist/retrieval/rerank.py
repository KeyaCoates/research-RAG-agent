"""LLM reranker: take the top N candidates from hybrid search and let a model put the passages
that best answer the question first. One model call per question, cached on disk so re-running
the eval never repeats (or pays for) the same call.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

from ragassist.ingestion.chunking import Chunk

PROMPT = """You are ranking passages for a question-answering system.
Question: {q}

Passages:
{passages}

Return ONLY a JSON list of passage numbers, best first, containing the {k} passages that most
directly answer the question. Example: [3, 0, 7]"""


def parse_ranking(text: str, n: int) -> list[int]:
    """Pull passage numbers out of the model's reply. Ignore anything invalid or repeated."""
    seen: list[int] = []
    for m in re.findall(r"\d+", text):
        i = int(m)
        if 0 <= i < n and i not in seen:
            seen.append(i)
    return seen


class LLMReranker:
    def __init__(self, llm, model: str, cache_path: str | pathlib.Path = "data/index/rerank_cache.json",
                 max_chars: int = 700):
        self.llm, self.model, self.max_chars = llm, model, max_chars
        self.cache_path = pathlib.Path(cache_path)
        self.cache: dict[str, list[int]] = (
            json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        )
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _key(self, query: str, cands: list[Chunk], k: int) -> str:
        raw = "|".join([self.model, query, str(k), *[c.id for c in cands]])
        return hashlib.sha256(raw.encode()).hexdigest()

    def rerank(self, query: str, cands: list[Chunk], k: int = 5) -> list[Chunk]:
        if len(cands) <= 1:
            return cands[:k]
        key = self._key(query, cands, k)
        if key not in self.cache:
            passages = "\n".join(f"[{i}] {c.text[: self.max_chars]}" for i, c in enumerate(cands))
            r = self.llm.generate(PROMPT.format(q=query, passages=passages, k=k), self.model)
            self.calls += 1
            self.tokens_in += r.tokens_in
            self.tokens_out += r.tokens_out
            self.cache[key] = parse_ranking(r.text, len(cands))
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        order = self.cache[key]
        # model's picks first, then the original hybrid order fills any gaps
        rest = [i for i in range(len(cands)) if i not in order]
        return [cands[i] for i in (order + rest)[:k]]
