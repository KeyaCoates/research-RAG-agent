from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def norm(s: str) -> str:
    """Collapse all whitespace so line breaks and spacing do not affect matching."""
    return " ".join(s.split())


def evidence_recall(quotes: list[str], chunk_texts: list[str]) -> float:
    """Fraction of evidence quotes that appear inside at least one retrieved chunk.
    This is chunk-level scoring: finding the right chapter is not enough,
    the retrieved text must actually contain the supporting passage."""
    if not quotes:
        return 0.0
    texts = [norm(t) for t in chunk_texts]
    found = sum(1 for q in quotes if any(norm(q) in t for t in texts))
    return found / len(quotes)


def evidence_reciprocal_rank(quotes: list[str], chunk_texts: list[str]) -> float:
    """1 / rank of the first retrieved chunk that contains any evidence quote."""
    qs = [norm(q) for q in quotes]
    for i, t in enumerate(chunk_texts, 1):
        nt = norm(t)
        if any(q in nt for q in qs):
            return 1.0 / i
    return 0.0
