"""Lexical (BM25) + dense retrieval fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from ragassist.ingestion.chunking import Chunk

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic offline embedder (feature hashing) for tests and CI smoke runs.
    Swap for VertexEmbedder in real runs."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tf = [Counter(d) for d in docs]
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            for q in query:
                if q in tf:
                    f = tf[q]
                    s[i] += self.idf[q] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vecs = embedder.embed([c.text for c in chunks])

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        """mode: 'bm25' | 'dense' | 'hybrid' (used for the ablation table)."""
        lex = np.argsort(-self.bm25.scores(tokenize(query)))[: max(k * 4, 20)].tolist()
        if hasattr(self.embedder, "embed_query"):  # real embedders treat queries differently from documents
            qvec = self.embedder.embed_query(query)
        else:
            qvec = self.embedder.embed([query])[0]
        dense_scores = self.vecs @ qvec
        den = np.argsort(-dense_scores)[: max(k * 4, 20)].tolist()
        if mode == "bm25":
            return [(self.chunks[i], 0.0) for i in lex[:k]]
        if mode == "dense":
            return [(self.chunks[i], float(dense_scores[i])) for i in den[:k]]
        return [(self.chunks[i], s) for i, s in rrf([lex, den])[:k]]
