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
