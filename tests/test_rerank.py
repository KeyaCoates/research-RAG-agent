"""Reranker tests: no network, no cost. A fake model stands in for Gemini."""
import pathlib
import tempfile

from ragassist.ingestion.chunking import Chunk
from ragassist.llm import LLMResult
from ragassist.retrieval.rerank import LLMReranker, parse_ranking


def chunks(n):
    return [Chunk(id=f"c{i}", doc_id="d", parent_id="p", section="", text=f"passage {i}") for i in range(n)]


class FakeModel:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, prompt, model):
        self.calls += 1
        return LLMResult(self.reply, 100, 5)


def test_parse_ignores_junk_duplicates_and_out_of_range():
    assert parse_ranking("Sure! [3, 0, 3, 99, 7]", 5) == [3, 0]
    assert parse_ranking("no numbers here", 5) == []


def test_model_picks_go_first_and_original_order_fills_gaps():
    with tempfile.TemporaryDirectory() as d:
        r = LLMReranker(FakeModel("[3, 1]"), "m", cache_path=pathlib.Path(d) / "c.json")
        out = r.rerank("q", chunks(6), k=4)
        assert [c.id for c in out] == ["c3", "c1", "c0", "c2"]


def test_bad_reply_falls_back_to_original_order():
    with tempfile.TemporaryDirectory() as d:
        r = LLMReranker(FakeModel("I cannot help"), "m", cache_path=pathlib.Path(d) / "c.json")
        assert [c.id for c in r.rerank("q", chunks(5), k=3)] == ["c0", "c1", "c2"]


def test_second_call_is_served_from_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = pathlib.Path(d) / "c.json"
        m = FakeModel("[2, 0]")
        LLMReranker(m, "m", cache_path=cache).rerank("q", chunks(4), k=2)
        LLMReranker(m, "m", cache_path=cache).rerank("q", chunks(4), k=2)  # new object, same cache file
        assert m.calls == 1
