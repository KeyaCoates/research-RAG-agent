"""Answer-level eval: baseline (hybrid + rerank + one answer call) versus the agentic pipeline.

Usage (PowerShell, GEMINI_API_KEY set):
    python -m ragassist.eval.answer_eval --corpus data/phak/corpus --golden data/phak/golden.jsonl \
        --unanswerable data/phak/golden_unanswerable.jsonl --out data/phak/results_agents.json

All model replies are cached in data/index/llm_cache.json, so a second run is free and identical.
Scoring:
  answerable questions   : a judge model says CORRECT or INCORRECT against the reference answer;
                           declining counts as wrongly declined.
  unanswerable questions : success = the system declines (it must not invent an answer).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from ragassist.agents.agentic import Models, agentic, baseline, is_decline
from ragassist.config import get_settings
from ragassist.eval.metrics import mean

JUDGE_PROMPT = """Question: {q}

Reference answer: {ref}

Candidate answer: {cand}

Does the candidate answer give the same key facts as the reference answer, without contradicting it? \
Extra detail is fine. Reply with one word: CORRECT or INCORRECT."""


def judge(llm, model: str, q: str, ref: str, cand: str) -> bool:
    out = llm.generate(JUDGE_PROMPT.format(q=q, ref=ref, cand=cand), model).text.strip().upper()
    return out.startswith("CORRECT")


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    return {
        "answerable_correct": round(mean([1.0 if r["correct"] else 0.0 for r in ans]), 3),
        "wrongly_declined": round(mean([1.0 if r["declined"] else 0.0 for r in ans]), 3),
        "unanswerable_declined": round(mean([1.0 if r["declined"] else 0.0 for r in un]), 3),
        "avg_llm_calls": round(mean([float(r["llm_calls"]) for r in rows]), 2),
        "avg_tokens_in": round(mean([float(r["tokens_in"]) for r in rows])),
        "avg_tokens_out": round(mean([float(r["tokens_out"]) for r in rows])),
        "n_answerable": len(ans),
        "n_unanswerable": len(un),
    }


def load_jsonl(path: pathlib.Path) -> list[dict]:
    lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, default="data/phak/corpus")
    ap.add_argument("--golden", type=pathlib.Path, default="data/phak/golden.jsonl")
    ap.add_argument("--unanswerable", type=pathlib.Path, default="data/phak/golden_unanswerable.jsonl")
    ap.add_argument("--out", type=pathlib.Path, default="data/phak/results_agents.json")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N answerable questions (quick test)")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 2
    from ragassist.eval.run_eval import load_index
    from ragassist.llm import CachedLLM, VertexLLM
    from ragassist.retrieval.embedders import VertexEmbedder
    from ragassist.retrieval.rerank import LLMReranker

    s = get_settings()
    llm = CachedLLM(VertexLLM())
    index = load_index(a.corpus, VertexEmbedder())
    reranker = LLMReranker(llm, s.router_model)
    models = Models(router=s.router_model, generator=s.generation_model, critic=s.generation_model)
    print(f"models: router={models.router} generator={models.generator}")

    golden = [g for g in load_jsonl(a.golden) if g.get("sources")]
    if a.limit:
        golden = golden[: a.limit]
    items = [(g, True) for g in golden] + [(u, False) for u in load_jsonl(a.unanswerable)]

    pipelines = {"baseline": baseline, "agentic": agentic}
    rows: dict[str, list[dict]] = {name: [] for name in pipelines}
    for i, (g, answerable) in enumerate(items, 1):
        for name, fn in pipelines.items():
            r = fn(g["question"], index, reranker, llm, models, k=a.k)
            declined = is_decline(r.answer)
            correct = False
            if answerable and not declined:
                correct = judge(llm, models.generator, g["question"], g["answer"], r.answer)
            rows[name].append({
                "question": g["question"], "answerable": answerable, "answer": r.answer,
                "declined": declined, "correct": correct, "attempts": r.attempts,
                "verdicts": r.verdicts, "llm_calls": r.usage.llm_calls,
                "tokens_in": r.usage.tokens_in, "tokens_out": r.usage.tokens_out,
            })
        if i % 10 == 0:
            print(f"  {i}/{len(items)} questions done", flush=True)

    results = {name: summarize(rs) for name, rs in rows.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": results, "rows": rows}, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    cols = [c for c in next(iter(results.values())) if not c.startswith("n_")]
    print(f"{'pipeline':9} " + " ".join(f"{c:22}" for c in cols))
    for name, r in results.items():
        print(f"{name:9} " + " ".join(f"{r[c]:<22}" for c in cols))
    print(f"(real model calls this run: {llm.api_calls}; the rest came from the cache)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
