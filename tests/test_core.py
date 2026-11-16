from ragassist.eval.metrics import recall_at_k, reciprocal_rank
from ragassist.ingestion.chunking import chunk_document, split_sections
from ragassist.pipeline import answer
from ragassist.retrieval.hybrid import HashingEmbedder, HybridIndex, rrf

DOC = "# Intro\nTransformers use attention.\n\n# Methods\nWe train with AdamW.\n\nLearning rate warmup helps."


def test_sections_and_parent_child():
    assert [t for t, _ in split_sections(DOC)] == ["Intro", "Methods"]
    chunks = chunk_document("paper1", DOC, max_chars=40)
    assert all(c.doc_id == "paper1" and c.parent_id.startswith("paper1#s") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_long_paragraph_is_not_cut_mid_sentence():
    sentences = [f"Sentence number {i} says something useful about flight." for i in range(30)]
    chunks = chunk_document("d", "# A\n" + " ".join(sentences), max_chars=200)
    assert len(chunks) > 1
    for s in sentences:
        assert any(s in c.text for c in chunks), f"sentence was split: {s}"


def test_evidence_scoring():
    from ragassist.eval.metrics import evidence_recall, evidence_reciprocal_rank

    texts = ["nothing here", "the  altimeter\nfreezes at the blockage altitude"]
    q = ["the altimeter freezes at the blockage altitude"]
    assert evidence_recall(q, texts) == 1.0
    assert evidence_reciprocal_rank(q, texts) == 0.5
    assert evidence_recall(["absent quote"], texts) == 0.0


def test_hash_changes_with_content():
    a = chunk_document("d", "# A\nhello")[0].meta["hash"]
    b = chunk_document("d", "# A\nhello world")[0].meta["hash"]
    assert a != b


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5


def test_rrf_prefers_agreement():
    assert rrf([[1, 2, 3], [3, 1, 2]])[0][0] == 1


def test_hybrid_and_pipeline_offline():
    chunks = chunk_document("p1", DOC) + chunk_document("p2", "# Data\nImageNet has 1000 classes.")
    idx = HybridIndex(chunks, HashingEmbedder())
    top = idx.search("how many classes in ImageNet", k=1)[0][0]
    assert top.doc_id == "p2"
    out = answer("how many classes in ImageNet", idx)
    assert "p2" in out["sources"] and out["trace"]["tokens_in"] > 0
