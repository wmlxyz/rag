import pytest
from rag.pipeline import Store, answer
from rag.errors import RagError
from rag.models import GenerationResult

class Embedder:
    def embed(self, texts):
        return [[1.0, 0.1] if "dense" in t else [0.1, 1.0] for t in texts]

class Generator:
    def generate(self, messages, **kwargs):
        return GenerationResult("Answer [1]", "fake")

def test_lexical_and_hybrid_are_versioned_and_rebuildable(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("dense unrelated", encoding="utf-8")
    (root / "b.txt").write_text("rareterm answer", encoding="utf-8")
    store = Store(tmp_path / "db")
    idx = store.build(root, Embedder(), embedding_id="v1", size=100, overlap=0)["index_id"]
    with pytest.raises(RagError, match="build-lexical"):
        store.search_lexical(idx, "rareterm")
    assert store.build_lexical(idx)["chunks"] == 2
    assert store.build_lexical(idx)["chunks"] == 2
    assert store.search_lexical(idx, '"rareterm" OR (', top_k=2)[0].source == "b.txt"
    result = answer(store, Embedder(), Generator(), index_id=idx, embedding_id="v1", question="dense rareterm", top_k=2, retrieval_mode="hybrid", persist=False)
    assert {h["source"] for h in result["retrieved"]} == {"a.txt", "b.txt"}
    assert result["score_type"] == "rrf" and result["retrieval_mode"] == "hybrid"
    with pytest.raises(ValueError):
        store.search_lexical("invalid; DROP TABLE chunks", "x")
