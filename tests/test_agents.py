"""Agent tests: a scripted fake model plays every role, so no network and no cost."""
import pathlib
import tempfile

from ragassist.agents.agentic import DECLINE, Models, agentic, baseline, is_decline, parse_json
from ragassist.eval.answer_eval import summarize
from ragassist.ingestion.chunking import chunk_document
from ragassist.llm import CachedLLM, LLMResult
from ragassist.retrieval.hybrid import HashingEmbedder, HybridIndex

MODELS = Models(router="r", generator="g", critic="c")


def make_index():
    chunks = chunk_document("ch1", "# Air\nThe altimeter freezes when the static port is blocked.") + \
        chunk_document("ch2", "# Rules\nClass C airspace has a five NM radius surface area.")
    return HybridIndex(chunks, HashingEmbedder())


class Scripted:
    """Replies depend on which role's prompt it sees. `critic` is a list of replies used in order."""

    def __init__(self, router, answers, critic):
        self.router, self.answers, self.critic = router, list(answers), list(critic)
        self.calls = []

    def generate(self, prompt, model):
        if prompt.startswith("You route"):
            kind, text = "router", self.router
        elif prompt.startswith("You check"):
            kind, text = "critic", self.critic.pop(0)
        else:
            kind, text = "answer", self.answers.pop(0)
        self.calls.append(kind)
        return LLMResult(text, 10, 5)


def test_parse_json_and_decline_detection():
    assert parse_json('Sure: {"verdict": "supported"} done') == {"verdict": "supported"}
    assert parse_json("no json") is None
    assert is_decline("I can’t answer that from the provided sources.")
    assert not is_decline("The altimeter freezes.")


def test_agent_accepts_a_supported_answer_in_one_attempt():
    llm = Scripted('{"route": "retrieve", "search_query": "static port blocked altimeter"}',
                   ["It freezes [ch1]."], ['{"verdict": "supported", "better_query": ""}'])
    r = agentic("What happens to the altimeter if the static port is blocked?", make_index(), None, llm, MODELS)
    assert r.answer == "It freezes [ch1]." and r.attempts == 1
    assert llm.calls == ["router", "answer", "critic"]


def test_agent_retries_with_a_better_query_then_succeeds():
    llm = Scripted('{"route": "retrieve", "search_query": "x"}',
                   [DECLINE, "Five NM radius [ch2]."],
                   ['{"verdict": "should_answer", "better_query": "class C radius"}',
                    '{"verdict": "supported", "better_query": ""}'])
    r = agentic("How wide is Class C airspace?", make_index(), None, llm, MODELS)
    assert r.attempts == 2 and r.verdicts == ["should_answer", "supported"]
    assert "class C radius" in r.queries and r.answer.startswith("Five NM")


def test_agent_declines_when_answer_stays_unsupported():
    llm = Scripted('{"route": "retrieve", "search_query": "x"}',
                   ["Made-up claim."],
                   ['{"verdict": "unsupported", "better_query": ""}'])
    r = agentic("What is the stall speed of a Cessna 172?", make_index(), None, llm, MODELS)
    assert r.answer == DECLINE


def test_router_can_decline_without_retrieval():
    llm = Scripted('{"route": "decline", "search_query": ""}', [], [])
    r = agentic("What is the capital of France?", make_index(), None, llm, MODELS)
    assert r.answer == DECLINE and llm.calls == ["router"]


def test_baseline_uses_one_answer_call():
    llm = Scripted("", ["It freezes [ch1]."], [])
    r = baseline("What happens if the static port is blocked?", make_index(), None, llm, MODELS)
    assert llm.calls == ["answer"] and r.usage.llm_calls == 1


def test_cached_llm_calls_the_model_once():
    with tempfile.TemporaryDirectory() as d:
        inner = Scripted("", ["a", "b"], [])
        c = CachedLLM(inner, path=pathlib.Path(d) / "c.json")
        assert c.generate("question one", "m").text == "a"
        assert c.generate("question one", "m").text == "a"  # from cache, not "b"
        assert CachedLLM(inner, path=pathlib.Path(d) / "c.json").generate("question one", "m").text == "a"
        assert inner.calls == ["answer"]


def test_summarize_counts_each_group_separately():
    rows = [
        {"answerable": True, "correct": True, "declined": False, "llm_calls": 3, "tokens_in": 100, "tokens_out": 10},
        {"answerable": True, "correct": False, "declined": True, "llm_calls": 3, "tokens_in": 100, "tokens_out": 10},
        {"answerable": False, "correct": False, "declined": True, "llm_calls": 3, "tokens_in": 100, "tokens_out": 10},
    ]
    s = summarize(rows)
    assert s["answerable_correct"] == 0.5 and s["wrongly_declined"] == 0.5 and s["unanswerable_declined"] == 1.0
