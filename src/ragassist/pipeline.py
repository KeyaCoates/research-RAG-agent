"""Baseline single-pass RAG. The agentic version (agents/) must beat this on the eval set."""
from __future__ import annotations

from ragassist.config import get_settings
from ragassist.llm import get_llm
from ragassist.retrieval.hybrid import HybridIndex
from ragassist.telemetry import RequestTrace

PROMPT = """Answer using ONLY the context. Cite sources as [doc_id]. If the context does not
contain the answer, say "I can't answer that from the provided sources."

QUESTION: {q}

CONTEXT:
{ctx}"""


def answer(question: str, index: HybridIndex, k: int = 5, mode: str = "hybrid") -> dict:
    s, llm, trace = get_settings(), get_llm(), RequestTrace()
    with trace.span("retrieve", mode=mode, k=k):
        hits = index.search(question, k=k, mode=mode)
    ctx = "\n".join(f"[{c.doc_id}] {c.text}" for c, _ in hits)
    with trace.span("generate", model=s.generation_model):
        r = llm.generate(PROMPT.format(q=question, ctx=ctx), s.generation_model)
    trace.record_llm(s.generation_model, r.tokens_in, r.tokens_out)
    return {
        "answer": r.text,
        "sources": sorted({c.doc_id for c, _ in hits}),
        "retrieved": [c.doc_id for c, _ in hits],
        "trace": trace.summary(),
    }
