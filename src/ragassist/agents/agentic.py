"""Multi-agent RAG, written as plain Python so it can be tested offline.

Roles (each is one focused model call):
  Router    - decides whether to answer and rewrites the question into a short search query.
  Retriever - hybrid search (+ optional reranker) over one or more queries.
  Answerer  - writes a short, cited answer from the passages only.
  Critic    - checks the answer is supported by the passages (self-reflection). If not, it proposes a
              better search query and the loop retries; if still unsupported, the system declines.

`baseline()` is the same retrieval and answerer with no router and no critic, so the comparison
isolates what the agent roles add.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ragassist.ingestion.chunking import Chunk

DECLINE = "I can't answer that from the provided sources."

ROUTER_PROMPT = """You route questions for an assistant that answers ONLY from chapters of the FAA Pilot's \
Handbook of Aeronautical Knowledge (principles of flight, aerodynamics, flight instruments, aircraft \
documents and maintenance rules, airspace).

QUESTION: {q}

Reply with JSON only: {{"route": "retrieve" or "decline", "search_query": "<the question rewritten as a short, \
keyword-rich search query>"}}
Use "decline" only if the question is clearly unrelated to aviation."""

ANSWER_PROMPT = """Answer the question using ONLY the passages. Be concise (1 to 3 sentences) and cite sources \
like [doc_id]. If the passages do not contain the answer, reply exactly: {decline}

QUESTION: {q}

PASSAGES:
{passages}"""

CRITIC_PROMPT = """You check an answer against its source passages.

QUESTION: {q}

DRAFT ANSWER: {a}

PASSAGES:
{passages}

Reply with JSON only: {{"verdict": "supported" or "unsupported" or "should_answer", \
"better_query": "<a different search query that might find the missing information, or empty>"}}
- supported: every claim in the answer is backed by the passages (or the answer declines and the passages \
really lack the answer).
- unsupported: the answer contains claims the passages do not back.
- should_answer: the answer declines, but the passages do contain the answer."""


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def is_decline(text: str) -> bool:
    t = text.replace("’", "'").lower()
    return "can't answer that from the provided sources" in t or "cannot answer that from the provided sources" in t


@dataclass
class Usage:
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, r) -> None:
        self.llm_calls += 1
        self.tokens_in += r.tokens_in
        self.tokens_out += r.tokens_out


@dataclass
class Models:
    router: str
    generator: str
    critic: str


@dataclass
class Result:
    answer: str
    sources: list[str]
    route: str
    attempts: int
    verdicts: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _ask(llm, prompt: str, model: str, usage: Usage) -> str:
    r = llm.generate(prompt, model)
    usage.add(r)
    return r.text


def _passages(hits: list[Chunk]) -> str:
    return "\n".join(f"[{c.doc_id}] {c.text[:900]}" for c in hits)


def retrieve(index, reranker, question: str, queries: list[str], usage: Usage,
             k: int = 5, n: int = 20) -> list[Chunk]:
    """Hybrid search for every query, merge without duplicates, then rerank against the original question."""
    seen: set[str] = set()
    cands: list[Chunk] = []
    for q in queries:
        for c, _ in index.search(q, k=n, mode="hybrid"):
            if c.id not in seen:
                seen.add(c.id)
                cands.append(c)
    cands = cands[: n + n // 2]
    if reranker is None:
        return cands[:k]
    usage.llm_calls += 1  # one reranking call (its tokens are not counted here)
    return reranker.rerank(question, cands, k=k)


def baseline(question: str, index, reranker, llm, models: Models, k: int = 5) -> Result:
    u = Usage()
    hits = retrieve(index, reranker, question, [question], u, k=k)
    ans = _ask(llm, ANSWER_PROMPT.format(decline=DECLINE, q=question, passages=_passages(hits)),
               models.generator, u).strip()
    return Result(ans, sorted({c.doc_id for c in hits}), "retrieve", 1, [], [question], u)


def agentic(question: str, index, reranker, llm, models: Models, k: int = 5, max_attempts: int = 3) -> Result:
    u = Usage()
    route = parse_json(_ask(llm, ROUTER_PROMPT.format(q=question), models.router, u)) or {}
    if route.get("route") == "decline":
        return Result(DECLINE, [], "decline", 0, [], [], u)
    queries = [question]
    sq = str(route.get("search_query") or "").strip()
    if sq and sq != question:
        queries.append(sq)

    verdicts: list[str] = []
    ans, hits = DECLINE, []
    for attempt in range(1, max_attempts + 1):
        hits = retrieve(index, reranker, question, queries, u, k=k)
        passages = _passages(hits)
        ans = _ask(llm, ANSWER_PROMPT.format(decline=DECLINE, q=question, passages=passages),
                   models.generator, u).strip()
        crit = parse_json(_ask(llm, CRITIC_PROMPT.format(q=question, a=ans, passages=passages),
                               models.critic, u)) or {}
        verdict = crit.get("verdict", "supported")
        if verdict == "should_answer" and not is_decline(ans):
            verdict = "supported"  # the critic contradicted itself; trust the answer it saw
        verdicts.append(verdict)
        if verdict == "supported":
            return Result(ans, sorted({c.doc_id for c in hits}), "retrieve", attempt, verdicts, queries, u)
        better = str(crit.get("better_query") or "").strip()
        if not better or better in queries:
            break  # nothing new to try
        queries.append(better)
    # still not supported after retries: decline instead of risking a made-up answer
    return Result(DECLINE, sorted({c.doc_id for c in hits}), "retrieve", len(verdicts), verdicts, queries, u)
