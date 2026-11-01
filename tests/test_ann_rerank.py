import json
import math
import pytest
from rag.config import ModelConfig
from rag.errors import ProviderError, ProviderResponseError, RagError
from rag.models import GenerationResult, RankedItem
from rag.pipeline import Store, answer
from rag.providers.dashscope_rerank import DashScopeReranker


class Embedder:
    def embed(self, texts):
        return [[1., .1] if "alpha" in t else [.1, 1.] for t in texts]


class Generator:
    def generate(self, messages, **kwargs):
        return GenerationResult("Answer [1]", "fake")


def test_rerank_validates_response_and_preserves_indexes(monkeypatch):
    adapter = DashScopeReranker(ModelConfig("dashscope_rerank", "https://example.test/api/v1", "secret", "m"))
    def good(path, payload):
        assert path.endswith("text-rerank/text-rerank")
        assert payload["input"]["query"] == "q"
        return {"output": {"results": [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1}]}, "usage": {"total_tokens": 10}}
    monkeypatch.setattr(adapter.client, "_post", good)
    assert [r.index for r in adapter.rerank("q", ["a", "b"])] == [1, 0]
    assert adapter.last_usage["total_tokens"] == 10
    monkeypatch.setattr(adapter.client, "_post", lambda *a: {"output": {"results": [{"index": 0, "relevance_score": math.nan}, {"index": 0, "relevance_score": 1}]}})
    with pytest.raises(ProviderResponseError): adapter.rerank("q", ["a", "b"])


def test_rerank_pipeline_and_explicit_fallback(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    store = Store(tmp_path / "db")
    idx = store.build(root, Embedder(), embedding_id="v1", size=100, overlap=0)["index_id"]
    class Reverse:
        def rerank(self, query, docs): return [RankedItem(1, .9), RankedItem(0, .1)]
    args = dict(index_id=idx, embedding_id="v1", question="alpha", top_k=1, candidate_k=2, persist=False)
    result = answer(store, Embedder(), Generator(), reranker=Reverse(), **args)
    assert result["evidence"][0]["source"] == "b.txt" and result["score_type"] == "rerank"
    assert len(result["candidates"]) == 2 and result["reranked"]
    class Failure:
        def rerank(self, query, docs): raise ProviderError("failure")
    with pytest.raises(ProviderError): answer(store, Embedder(), Generator(), reranker=Failure(), **args)
    result = answer(store, Embedder(), Generator(), reranker=Failure(), rerank_fallback=True, **args)
    assert result["evidence"][0]["source"] == "a.txt" and result["rerank_error"]


def test_ann_persistence_and_corruption_detection(tmp_path):
    pytest.importorskip("faiss")
    from rag.ann import build_ann
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    path = tmp_path / "db"
    store = Store(path)
    idx = store.build(root, Embedder(), embedding_id="v1", size=100, overlap=0)["index_id"]
    meta = build_ann(store, idx)
    store = Store(path)
    exact = store.search(idx, "alpha", Embedder(), embedding_id="v1", top_k=2)
    ann = store.search(idx, "alpha", Embedder(), embedding_id="v1", top_k=2, backend="ann")
    assert [h.chunk_id for h in exact] == [h.chunk_id for h in ann]
    assert all(abs(a.score-b.score) < 1e-5 for a, b in zip(exact, ann))
    ann_path = tmp_path / "db.ann" / meta["filename"]
    ann_path.write_bytes(b"broken")
    with pytest.raises(RagError, match="checksum"):
        Store(path).search(idx, "alpha", Embedder(), embedding_id="v1", backend="ann")
