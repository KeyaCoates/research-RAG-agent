"""Eval harness. Usage:
    python -m ragassist.eval.run_eval --corpus data/eval/corpus --golden data/eval/golden.jsonl \
        --modes bm25 dense hybrid --min-recall 0.6

Exits non-zero if the best mode's recall@k falls below --min-recall, so CI
(Azure Pipelines) fails the build on retrieval regressions.
Golden format (jsonl): {"question": ..., "answer": ..., "sources": ["doc_id", ...]}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from ragassist.eval.metrics import mean, recall_at_k, reciprocal_rank
from ragassist.ingestion.chunking import chunk_document
from ragassist.retrieval.hybrid import HashingEmbedder, HybridIndex


def load_index(corpus: pathlib.Path) -> HybridIndex:
    chunks = []
    for p in sorted(corpus.glob("*.md")):
        chunks += chunk_document(p.stem, p.read_text(encoding="utf-8"))
    return HybridIndex(chunks, HashingEmbedder())


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
    golden = [
        json.loads(line)
        for line in a.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {}
    for mode in a.modes:
        rec, mrr = [], []
        for g in golden:
            ids = [c.doc_id for c, _ in index.search(g["question"], k=a.k, mode=mode)]
            rel = set(g["sources"])
            rec.append(recall_at_k(ids, rel, a.k))
            mrr.append(reciprocal_rank(ids, rel))
        results[mode] = {f"recall@{a.k}": round(mean(rec), 3), "mrr": round(mean(mrr), 3), "n": len(golden)}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{'mode':8} {'recall@'+str(a.k):10} mrr")
    for m, r in results.items():
        print(f"{m:8} {r[f'recall@{a.k}']:<10} {r['mrr']}")
    best = max(r[f"recall@{a.k}"] for r in results.values())
    if best < a.min_recall:
        print(f"FAIL: best recall@{a.k} {best} < {a.min_recall}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
