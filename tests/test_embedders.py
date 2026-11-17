"""Tests the Vertex embedder's batching and cache without calling Google (no network, no cost)."""
import pathlib
import tempfile

import numpy as np

from ragassist.retrieval.embedders import VertexEmbedder


class FakeVertex(VertexEmbedder):
    """Overrides only the network call. Everything else (cache, batching, normalizing) is real."""

    def __init__(self, **kw):
        self.calls = []
        super().__init__(client=object(), **kw)

    def _call(self, texts, task):
        self.calls.append((task, list(texts)))
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_batching_and_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "cache.json"
        e = FakeVertex(cache_path=cache, batch_size=2)
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        v1 = e.embed(texts)
        assert len(e.calls) == 3  # 5 texts in batches of 2 -> 3 calls
        assert np.allclose(np.linalg.norm(v1, axis=1), 1.0)  # vectors are normalized
        e.embed(texts)
        assert len(e.calls) == 3  # second run is served entirely from the cache
        e2 = FakeVertex(cache_path=cache, batch_size=2)  # new object, same cache file
        e2.embed(texts)
        assert e2.calls == []


def test_query_and_document_use_different_task_types():
    with tempfile.TemporaryDirectory() as d:
        e = FakeVertex(cache_path=pathlib.Path(d) / "c.json")
        e.embed(["same text"])
        e.embed_query("same text")
        assert [t for t, _ in e.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


def test_non_ascii_text_survives_the_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "c.json"
        FakeVertex(cache_path=cache).embed(["59 °F “quoted”"])
        e2 = FakeVertex(cache_path=cache)
        e2.embed(["59 °F “quoted”"])
        assert e2.calls == []


def test_retry_delay_follows_rate_limit_message():
    from ragassist.retrieval.embedders import retry_delay

    msg = "429 RESOURCE_EXHAUSTED. Please retry in 45.8s."
    assert abs(retry_delay(msg, 0) - 46.8) < 0.01
    assert retry_delay("429 RESOURCE_EXHAUSTED", 0) == 60.0
    assert retry_delay("500 server error", 2) == 4.0


def test_throttle_waits_when_over_the_per_minute_limit():
    from ragassist.retrieval import embedders

    waits = []
    original = embedders._sleep
    embedders._sleep = lambda s: (waits.append(s), e._sent.clear())  # pretend the minute passed
    try:
        with tempfile.TemporaryDirectory() as d:
            e = FakeVertex(cache_path=pathlib.Path(d) / "c.json", batch_size=4, max_per_minute=8)
            e.embed([f"text {i}" for i in range(12)])  # 12 texts, limit 8 per minute
    finally:
        embedders._sleep = original
    assert len(waits) >= 1
    assert len(e.calls) == 3


def test_cache_is_saved_after_each_batch():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "c.json"

        class Crashes(FakeVertex):
            def _call(self, texts, task):
                if len(self.calls) == 1:
                    raise RuntimeError("limit hit")
                return super()._call(texts, task)

        e = Crashes(cache_path=cache, batch_size=2)
        try:
            e.embed(["a", "b", "c", "d"])
        except RuntimeError:
            pass
        e2 = FakeVertex(cache_path=cache, batch_size=2)
        e2.embed(["a", "b", "c", "d"])
        assert len(e2.calls) == 1  # only the second batch had to be redone
