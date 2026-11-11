"""FastAPI service (Cloud Run). Run: uvicorn ragassist.api:app --port 8080"""
from __future__ import annotations

import pathlib

from fastapi import FastAPI
from pydantic import BaseModel

from ragassist.eval.run_eval import load_index
from ragassist.pipeline import answer

app = FastAPI(title="RAG assistant")
_index = None


def index():
    global _index
    if _index is None:
        _index = load_index(pathlib.Path("data/eval/corpus"))  # TODO: load from Vector Search / pgvector
    return _index


class Ask(BaseModel):
    question: str
    k: int = 5


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask):
    return answer(body.question, index(), k=body.k)
