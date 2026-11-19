"""Builds the question-answering service from a saved index, so the web server starts in seconds
and only embeds the user's question (never the whole corpus)."""
from __future__ import annotations

import pathlib
import time

import numpy as np

from ragassist.agents.agentic import Models, agentic, baseline
from ragassist.ingestion.chunking import Chunk, chunk_document
from ragassist.retrieval.hybrid import HybridIndex


def load_chunks(corpus: pathlib.Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(pathlib.Path(corpus).glob("*.md")):
        chunks += chunk_document(p.stem, p.read_text(encoding="utf-8"))
    return chunks


def save_index(corpus: pathlib.Path, vecs: np.ndarray, out: pathlib.Path) -> None:
    ids = np.array([c.id for c in load_chunks(corpus)])
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vecs=vecs.astype(np.float32), ids=ids)


def load_index_file(corpus: pathlib.Path, index_file: pathlib.Path, embedder) -> HybridIndex:
    """Load chunks and saved vectors. Refuses to start if the corpus changed since the index was built."""
    chunks = load_chunks(corpus)
    z = np.load(index_file)
    if list(z["ids"]) != [c.id for c in chunks]:
        raise ValueError("index file does not match the corpus; run scripts/build_index.py again")
    return HybridIndex(chunks, embedder, vecs=z["vecs"])


class Service:
    def __init__(self, index: HybridIndex, reranker, llm, models: Models):
        self.index, self.reranker, self.llm, self.models = index, reranker, llm, models

    def ask(self, question: str, mode: str = "agentic", k: int = 5) -> dict:
        fn = agentic if mode == "agentic" else baseline
        t0 = time.perf_counter()
        r = fn(question, self.index, self.reranker, self.llm, self.models, k=k)
        seconds = time.perf_counter() - t0
        return {
            "answer": r.answer,
            "sources": r.sources,
            "mode": mode,
            "attempts": r.attempts,
            "llm_calls": r.usage.llm_calls,
            "tokens_in": r.usage.tokens_in,
            "tokens_out": r.usage.tokens_out,
            "latency_ms": round(seconds * 1000),
        }


def build_service(corpus: str = "data/phak/corpus", index_file: str = "data/phak/index.npz",
                  cache_dir: str = "/tmp") -> Service:
    """Real service: Gemini embeddings for questions, Gemini models for the agents. Needs GEMINI_API_KEY."""
    from ragassist.config import get_settings
    from ragassist.llm import CachedLLM, VertexLLM
    from ragassist.retrieval.embedders import VertexEmbedder
    from ragassist.retrieval.rerank import LLMReranker

    s = get_settings()
    llm = CachedLLM(VertexLLM(), path=f"{cache_dir}/llm_cache.json")
    embedder = VertexEmbedder(cache_path=f"{cache_dir}/embedding_cache.json")
    index = load_index_file(pathlib.Path(corpus), pathlib.Path(index_file), embedder)
    reranker = LLMReranker(llm, s.router_model, cache_path=f"{cache_dir}/rerank_cache.json")
    return Service(index, reranker, llm, Models(router=s.router_model, generator=s.generation_model,
                                                critic=s.generation_model))
