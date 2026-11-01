import pytest
from rag.errors import ProviderError
from rag.models import GenerationResult
from rag.pipeline import IndexError, Store, answer, chunk_text


class Embedding:
    def embed(self, texts):
        return [[1.0, 0.0] if "苹果" in text else [0.0, 1.0] for text in texts]


class Generation:
    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if "Rewrite" in messages[0].content:
            return GenerationResult("苹果的颜色", "fake")
        return GenerationResult("苹果是红色的。[1]", "fake")


def test_chunk_offsets_and_validation():
    assert chunk_text("abcdef", 4, 1) == [(0, 4, "abcd"), (3, 6, "def")]
    with pytest.raises(ValueError):
        chunk_text("abc", 2, 2)


def test_build_search_session_and_rewrite(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "fruit.txt").write_text("苹果是红色的。", encoding="utf-8")
    (docs / "other.md").write_text("天空是蓝色的。", encoding="utf-8")
    db = tmp_path / "rag.sqlite"
    store = Store(db)
    index = store.build(docs, Embedding(), embedding_id="test-v1", size=20, overlap=0)
    assert store.build(docs, Embedding(), embedding_id="test-v1", size=20, overlap=0)["index_id"] == index["index_id"]
    store = Store(db)  # Reload from disk, not in-memory state.
    generator = Generation()
    first = answer(store, Embedding(), generator, index_id=index["index_id"], embedding_id="test-v1", question="苹果是什么颜色？", session_id="session", top_k=1)
    assert first["evidence"][0]["source"] == "fruit.txt"
    assert first["has_valid_citation"] and not first["invalid_citations"]
    second = answer(store, Embedding(), generator, index_id=index["index_id"], embedding_id="test-v1", question="它是什么颜色？", session_id="session", strategy="rewrite", top_k=1)
    assert second["query"] == "苹果的颜色"
    assert len(store.history("session")) == 4
    with pytest.raises(IndexError, match="identity"):
        store.search(index["index_id"], "苹果", Embedding(), embedding_id="other")


def test_failure_does_not_publish_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.txt").write_text("some text", encoding="utf-8")
    store = Store(tmp_path / "db.sqlite")
    class BadEmbedding:
        def embed(self, texts):
            raise ProviderError("offline")
    with pytest.raises(ProviderError):
        store.build(docs, BadEmbedding(), embedding_id="v1")
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM indexes").fetchone()[0] == 0


def test_batched_staging_cleans_up_after_midway_failure(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.txt").write_text("abcdefghij", encoding="utf-8")
    store = Store(tmp_path / "db.sqlite")
    class FailSecondBatch:
        def __init__(self): self.calls = 0
        def embed(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise ProviderError("offline")
            return [[1.0] for _ in texts]
    with pytest.raises(ProviderError):
        store.build(docs, FailSecondBatch(), embedding_id="v1", size=2, overlap=0, batch_size=2)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM indexes").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert not list(tmp_path.glob("rag-index-stage-*.sqlite"))


def test_no_evidence_and_invalid_citation(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.txt").write_text("苹果", encoding="utf-8")
    store = Store(tmp_path / "db.sqlite")
    idx = store.build(docs, Embedding(), embedding_id="v1", size=10, overlap=0)["index_id"]
    gen = Generation()
    result = answer(store, Embedding(), gen, index_id=idx, embedding_id="v1", question="天气", min_score=0.5)
    assert not result["evidence"] and not gen.calls
    assert "无法" in result["answer"]
    class InvalidCitation(Generation):
        def generate(self, messages, **kwargs):
            return GenerationResult("未知[99]", "fake")
    result = answer(store, Embedding(), InvalidCitation(), index_id=idx, embedding_id="v1", question="苹果", citation_policy="warn")
    assert result["invalid_citations"] == [99]


def test_rewrite_failure_falls_back(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.txt").write_text("苹果", encoding="utf-8")
    store = Store(tmp_path / "db.sqlite")
    idx = store.build(docs, Embedding(), embedding_id="v1", size=10, overlap=0)["index_id"]
    store.save_turn("s", "苹果是什么？", "一种水果")
    class FailedRewrite(Generation):
        def generate(self, messages, **kwargs):
            if "Rewrite" in messages[0].content:
                raise ProviderError("unavailable")
            return super().generate(messages, **kwargs)
    result = answer(store, Embedding(), FailedRewrite(), index_id=idx, embedding_id="v1", session_id="s", question="苹果颜色？", strategy="rewrite")
    assert result["query"] == "苹果颜色？" and result["rewrite_error"]
