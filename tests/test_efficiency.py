import pytest
from rag.cache import CachedEmbedder
from rag.models import GenerationResult
from rag.pipeline import Hit, Store, answer, uncovered_spans


def test_lru_cache_batch_duplicates_copy_and_eviction():
    class Provider:
        def __init__(self): self.calls = []
        def embed(self, texts):
            self.calls.append(list(texts))
            return [[float(len(t))] for t in texts]
    provider = Provider()
    cache = CachedEmbedder(provider, capacity=2)
    assert cache.embed(["a", "bb", "a"]) == [[1], [2], [1]]
    result = cache.embed(["a"])
    result[0][0] = -1
    assert cache.embed(["a"]) == [[1]]
    cache.embed(["ccc"])
    cache.embed(["bb"])
    assert provider.calls == [["a", "bb"], ["ccc"], ["bb"]]
    assert cache.stats()["provider_calls"] == 3 and cache.stats()["size"] == 2


def test_cache_does_not_store_failed_fetches():
    class Provider:
        def __init__(self): self.calls = 0
        def embed(self, texts):
            self.calls += 1
            if self.calls == 1: raise ValueError("failed")
            return [[1.0] for _ in texts]
    p = Provider()
    c = CachedEmbedder(p)
    with pytest.raises(ValueError): c.embed(["x"])
    assert c.embed(["x"]) == [[1.0]] and p.calls == 2


def test_overlap_uses_only_actual_included_offsets():
    hit = Hit("id", "doc", 50, 150, "x" * 100, 1.0)
    assert uncovered_spans(hit, [{"source": "doc", "start": 0, "end": 110}]) == [(110, 150)]
    assert uncovered_spans(hit, [{"source": "doc", "start": 0, "end": 90}]) == [(90, 150)]
    assert uncovered_spans(hit, [{"source": "other", "start": 50, "end": 150}]) == [(50, 150)]
    assert uncovered_spans(hit, [{"source": "doc", "start": 50, "end": 90}, {"source": "doc", "start": 80, "end": 110}]) == [(110, 150)]
    assert uncovered_spans(hit, [{"source": "doc", "start": 70, "end": 100}]) == [(50, 70), (100, 150)]


def test_dedup_context_retains_unique_offsets(tmp_path):
    class Embedding:
        def embed(self, texts): return [[1.0] for _ in texts]
    class Generator:
        def generate(self, messages, **kwargs): return GenerationResult("OK [1]", "fake")
    root = tmp_path / "docs"
    root.mkdir()
    text = "abcdefghijklmnop"
    (root / "a.txt").write_text(text, encoding="utf-8")
    store = Store(tmp_path / "db")
    idx = store.build(root, Embedding(), embedding_id="v1", size=8, overlap=4)["index_id"]
    args = dict(index_id=idx, embedding_id="v1", question="query", top_k=10, persist=False)
    standard = answer(store, Embedding(), Generator(), **args)
    dedup = answer(store, Embedding(), Generator(), context_mode="deduplicate", **args)
    assert dedup["context_chars_used"] == len(text)
    assert standard["context_chars_used"] > dedup["context_chars_used"]
    assert all(e["text"] == text[e["start"]:e["end"]] for e in dedup["evidence"])
