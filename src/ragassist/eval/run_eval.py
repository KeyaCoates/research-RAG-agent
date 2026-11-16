"""Eval harness. Usage:
    python -m ragassist.eval.run_eval --corpus data/phak/corpus --golden data/phak/golden.jsonl \
        --modes bm25 dense hybrid --min-recall 0.6

Scores two ways:
  doc_*    : did we retrieve a chunk from the right document? (easy, inflated for small corpora)
  chunk_*  : does a retrieved chunk actually contain the evidence quote? (stricter, use this one)

Exits non-zero if the best mode's primary recall falls below --min-recall, so CI
(Azure Pipelines) fails the build on retrieval regressions. The primary metric is
chunk-level when the golden file has "evidence" quotes, otherwise document-level.

Golden format (jsonl):
  {"question": ..., "answer": ..., "sources": ["doc_id"], "evidence": [{"source": "doc_id", "quote": "..."}]}
Entries with no sources (unanswerable questions) are skipped here; they need an abstain metric.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from ragassist.eval.metrics import (
    evidence_recall,
    evidence_reciprocal_rank,
    mean,
    recall_at_k,
    reciprocal_rank,
)
from ragassist.ingestion.chunking import chunk_document
from ragassist.retrieval.hybrid import HashingEmbedder, HybridIndex


def load_index(corpus: pathlib.Path) -> HybridIndex:
    chunks = []
    for p in sorted(pathlib.Path(corpus).glob("*.md")):
        chunks += chunk_document(p.stem, p.read_text(encoding="utf-8"))
    return HybridIndex(chunks, HashingEmbedder())


def load_golden(path: pathlib.Path) -> list[dict]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [g for g in (json.loads(l) for l in lines if l.strip()) if g.get("sources")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default="data/eval/corpus")
    ap.add_argument("--golden", type=pathlib.Path, default="data/eval/golden.jsonl")
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.0)
    ap.add_argument("--out", type=pathlib.Path, default="eval-results/results.json")
    a = ap.parse_args()

    index = load_index(a.corpus)
    golden = load_golden(a.golden)
    has_evidence = all(g.get("evidence") for g in golden)
    results = {}
    for mode in a.modes:
        d_rec, d_mrr, c_rec, c_mrr = [], [], [], []
        for g in golden:
            hits = [c for c, _ in index.search(g["question"], k=a.k, mode=mode)]
            ids = [c.doc_id for c in hits]
            rel = set(g["sources"])
            d_rec.append(recall_at_k(ids, rel, a.k))
            d_mrr.append(reciprocal_rank(ids, rel))
            if has_evidence:
                quotes = [e["quote"] for e in g["evidence"]]
                texts = [c.text for c in hits]
                c_rec.append(evidence_recall(quotes, texts))
                c_mrr.append(evidence_reciprocal_rank(quotes, texts))
        row = {f"doc_recall@{a.k}": round(mean(d_rec), 3), "doc_mrr": round(mean(d_mrr), 3), "n": len(golden)}
        if has_evidence:
            row[f"chunk_recall@{a.k}"] = round(mean(c_rec), 3)
            row["chunk_mrr"] = round(mean(c_mrr), 3)
        results[mode] = row

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    cols = list(next(iter(results.values())).keys())
    cols.remove("n")
    print(f"{'mode':8} " + " ".join(f"{c:16}" for c in cols))
    for m, r in results.items():
        print(f"{m:8} " + " ".join(f"{r[c]:<16}" for c in cols))

    primary = f"chunk_recall@{a.k}" if has_evidence else f"doc_recall@{a.k}"
    best = max(r[primary] for r in results.values())
    if best < a.min_recall:
        print(f"FAIL: best {primary} {best} < {a.min_recall}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
