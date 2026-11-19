"""FastAPI service (Cloud Run). Run locally: uvicorn ragassist.api:app --port 8080
Needs GEMINI_API_KEY and data/phak/index.npz (build it with scripts/build_index.py)."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ragassist.service import build_service

app = FastAPI(title="RAG assistant: FAA handbook Q&A")
_service = None
_hits: dict[str, deque] = defaultdict(deque)
LIMIT_PER_MINUTE = 6  # keeps a public demo from burning the model quota


def service():
    global _service
    if _service is None:
        _service = build_service()
    return _service


def check_rate_limit(client: str, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    q = _hits[client]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMIT_PER_MINUTE:
        return False
    q.append(now)
    return True


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field(default="agentic", pattern="^(agentic|baseline)$")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask, request: Request):
    client = request.client.host if request.client else "unknown"
    if not check_rate_limit(client):
        raise HTTPException(status_code=429, detail="Too many requests. Try again in a minute.")
    return service().ask(body.question, mode=body.mode)
