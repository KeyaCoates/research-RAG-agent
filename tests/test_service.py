"""Service tests: saved index round trip and rate limiting. No network."""
import pathlib
import tempfile

import numpy as np

from ragassist.agents.agentic import Models
from ragassist.llm import LLMResult
from ragassist.retrieval.hybrid import HashingEmbedder
from ragassist.service import Service, load_chunks, load_index_file, save_index


def make_corpus(d):
    (d / "a.md").write_text("# Air\nThe altimeter freezes when the static port is blocked.", encoding="utf-8")
    (d / "b.md").write_text("# Rules\nClass C airspace has a five NM radius.", encoding="utf-8")


def test_index_round_trip_and_mismatch_is_refused():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        make_corpus(t)
        emb = HashingEmbedder()
        vecs = emb.embed([c.text for c in load_chunks(t)])
        save_index(t, vecs, t / "index.npz")
        idx = load_index_file(t, t / "index.npz", emb)
        assert np.allclose(idx.vecs, vecs)
        assert idx.search("static port blocked", k=1)[0][0].doc_id == "a"
        (t / "c.md").write_text("# New\nA new document appears.", encoding="utf-8")  # corpus changed
        try:
            load_index_file(t, t / "index.npz", emb)
            raise AssertionError("should have refused a stale index")
        except ValueError:
            pass


class Scripted:
    def generate(self, prompt, model):
        return LLMResult("It freezes [a]." if "PASSAGES" in prompt else "{}", 10, 5)


def test_service_reports_cost_and_latency():
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t)
        make_corpus(t)
        emb = HashingEmbedder()
        save_index(t, emb.embed([c.text for c in load_chunks(t)]), t / "i.npz")
        s = Service(load_index_file(t, t / "i.npz", emb), None, Scripted(), Models("r", "g", "c"))
        out = s.ask("What happens if the static port is blocked?", mode="baseline")
        assert out["answer"].startswith("It freezes") and out["llm_calls"] == 1
        assert out["latency_ms"] >= 0 and out["tokens_in"] == 10
