"""Real embeddings from Vertex AI, with batching, retries and an on-disk cache.

The cache means re-running the eval never pays for (or waits for) the same text twice.
Run `pip install -e ".[gcp]"` and sign in with `gcloud auth application-default login` first.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time

import numpy as np

from ragassist.config import get_settings


def _sleep(seconds: float) -> None:  # separate function so tests can replace it
    time.sleep(seconds)


def retry_delay(error_text: str, attempt: int) -> float:
    """How long to wait after a failed call. Rate-limit errors (429) usually say how long to wait."""
    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
        m = re.search(r"retry in ([\d.]+)s", error_text)
        return float(m.group(1)) + 1.0 if m else 60.0
    return float(2**attempt)


class VertexEmbedder:
    def __init__(
        self,
        client=None,
        model: str | None = None,
        cache_path: str | pathlib.Path = "data/index/embedding_cache.json",
        batch_size: int = 16,
        max_per_minute: int | None = None,
    ):
        s = get_settings()
        api_key = os.getenv("GEMINI_API_KEY")  # free Google AI Studio key, no Google Cloud project needed
        default_model = "gemini-embedding-001" if api_key else s.embedding_model
        self.model = model or os.getenv("EMBEDDING_MODEL") or default_model
        client_given = client is not None
        if client is None:
            from google import genai  # optional dependency, only needed for real runs

            if api_key:
                client = genai.Client(api_key=api_key)
            else:
                client = genai.Client(vertexai=True, project=s.gcp_project, location=s.gcp_region)
        # The free Gemini API key allows 100 embedding requests per minute, so stay under that.
        if max_per_minute is None and api_key and not client_given:
            max_per_minute = 90
        self.max_per_minute = max_per_minute
        self._sent: list[float] = []  # times of recent requests, used to stay under the limit
        self.client = client
        self.batch_size = batch_size
        self.cache_path = pathlib.Path(cache_path)
        self.cache: dict[str, list[float]] = self._load()
        self.api_calls = 0

    # cache ---------------------------------------------------------------
    def _load(self) -> dict:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _key(self, text: str, task: str) -> str:
        return hashlib.sha256(f"{self.model}|{task}|{text}".encode()).hexdigest()

    # api -----------------------------------------------------------------
    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        for attempt in range(6):
            try:
                r = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type=task),
                )
                self.api_calls += 1
                return [list(e.values) for e in r.embeddings]
            except Exception as e:
                if attempt == 5:
                    raise
                _sleep(retry_delay(str(e), attempt))  # rate limits tell us how long to wait
        raise RuntimeError("unreachable")

    def _throttle(self, n: int) -> None:
        """Wait, if needed, so that no more than max_per_minute texts are sent in any 60 seconds."""
        if not self.max_per_minute:
            return
        while True:
            now = time.monotonic()
            self._sent = [s for s in self._sent if now - s < 60]
            if len(self._sent) + n <= self.max_per_minute:
                break
            _sleep(60 - (now - self._sent[0]) + 0.5)
        self._sent.extend([time.monotonic()] * n)

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        keys = [self._key(t, task) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        for start in range(0, len(missing), self.batch_size):
            idx = missing[start : start + self.batch_size]
            self._throttle(len(idx))
            vecs = self._call([texts[i] for i in idx], task)
            for i, v in zip(idx, vecs):
                self.cache[keys[i]] = v
            self._save()  # save after every batch, so a crash or limit error loses nothing
        arr = np.array([self.cache[k] for k in keys], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    # public --------------------------------------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed documents (chunks)."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Queries and documents use different task types on purpose."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]
