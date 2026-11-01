import json
import pytest
from rag.evaluation import answer_scores, map_sources, prepare_qrecc, run_experiment
from rag.models import GenerationResult
from rag.pipeline import Store


class Embedding:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Generation:
    def generate(self, messages, **kwargs):
        if "Rewrite" in messages[0].content:
            return GenerationResult("Standalone question", "fake")
        return GenerationResult("red [1]", "fake")


def test_qrecc_deterministic_and_context(tmp_path):
    rows = [{"Conversation_no": 1, "Turn_no": 1, "Question": "What?", "Context": [], "Rewrite": "What?", "Answer": "red", "Answer_URL": "https://example.org"},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "Why?", "Context": ["What?", "red"], "Rewrite": "Why red?", "Answer": "reason"},
            {"Conversation_no": 2, "Turn_no": 1, "Question": "Who?", "Rewrite": "Who?", "Context": []}]
    path = tmp_path / "qrecc.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    a = prepare_qrecc(path, split="test", conversations=2, seed=42)
    assert a == prepare_qrecc(path, split="test", conversations=2, seed=42)
    assert a["records"][1]["history"][1] == {"role": "assistant", "content": "red"}
    assert a["records"][1]["relevant_sources"] == []


def test_qrecc_missing_answer_and_rewritten_context(tmp_path):
    rows = [{"Conversation_no": 1, "Turn_no": 1, "Question": "It?", "Context": [], "Rewrite": "What is it?", "Answer": ""},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "Why?", "Context": ["What is it?"], "Rewrite": "Why is it?", "Answer": "Yes"}]
    path = tmp_path / "q.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    dataset = prepare_qrecc(path, split="test", conversations=1, seed=1)
    assert dataset["records"][1]["history"] == [{"role": "user", "content": "What is it?"}]


def test_supplemental_truth_requires_matching_rewrite_and_answer(tmp_path):
    rows = [{"Conversation_no": 1, "Turn_no": 1, "Question": "What?", "Context": [], "Rewrite": "What?", "Answer": "a"},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "Why?", "Context": ["What?", "a"], "Rewrite": "Why?", "Answer": "b"}]
    truth = [{"Conversation_no": 1, "Turn_no": 1, "Truth_rewrite": "What?", "Truth_answer": "a", "Truth_passages": ["p1"]},
             {"Conversation_no": 1, "Turn_no": 2, "Truth_rewrite": "different", "Truth_answer": "b", "Truth_passages": ["p2"]}]
    p, t = tmp_path / "data.json", tmp_path / "truth.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    t.write_text(json.dumps(truth), encoding="utf-8")
    records = prepare_qrecc(p, split="train", conversations=1, seed=42, ground_truth_path=t)["records"]
    assert records[0]["reference_passages"] == ["p1"]
    assert records[1]["reference_passages"] == [] and not records[1]["supplemental_truth_verified"]


def test_inconsistent_history_conversation_excluded_as_a_whole(tmp_path):
    rows = [{"Conversation_no": 1, "Turn_no": 1, "Question": "What?", "Context": [], "Rewrite": "What?", "Answer": "a"},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "Why?", "Context": ["What?", "different"], "Rewrite": "Why?", "Answer": "b"},
            {"Conversation_no": 2, "Turn_no": 1, "Question": "Who?", "Context": [], "Rewrite": "Who?", "Answer": "c"}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    output = prepare_qrecc(path, split="train", conversations=2, seed=42, skip_invalid_conversations=True)
    assert output["excluded_context_mismatch_conversations"] == ["1"]
    assert [r["conversation_id"] for r in output["records"]] == ["train:2"]


def test_scoring_and_fixed_history(tmp_path):
    assert answer_scores("red [1]", "red")["exact_match"] == 1
    assert answer_scores("any", "") is None
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.txt").write_text("red", encoding="utf-8")
    store = Store(tmp_path / "db")
    index = store.build(docs, Embedding(), embedding_id="v1", size=100, overlap=0)["index_id"]
    record = {"id": "1", "conversation_id": "c", "question": "Why?", "history": [{"role": "user", "content": "What?"}, {"role": "assistant", "content": "red"}], "human_rewrite": "Why red?", "reference_answer": "red", "relevant_sources": ["one.txt"]}
    run = run_experiment(store, Embedding(), Generation(), {"records": [record]}, index_id=index, embedding_id="v1", strategies=["current", "history", "rewrite", "human"], options={"top_k": 1})
    assert [r["query"] for r in run["results"]] == ["Why?", "user: What?\nassistant: red\nuser: Why?", "Standalone question", "Why red?"]
    assert all(r["metrics"]["retrieval_hit"] == 1 for r in run["results"])
    assert store.history("evaluation") == []
    assert run["summary"]["current"]["answer_token_f1"]["mean"] == 1


def test_unlabeled_retrieval_not_scored(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.txt").write_text("red", encoding="utf-8")
    store = Store(tmp_path / "db")
    index = store.build(docs, Embedding(), embedding_id="v1", size=100, overlap=0)["index_id"]
    record = {"id": "1", "conversation_id": "c", "question": "What?", "history": [], "reference_answer": "red", "relevant_sources": []}
    run = run_experiment(store, Embedding(), Generation(), {"records": [record]}, index_id=index, embedding_id="v1", strategies=["current"], options={})
    assert run["summary"]["current"]["document_hit_at_k"] == {"mean": None, "n": 0}


def test_url_mapping_only_exact_and_indexed():
    records = {"records": [{"reference_url": "https://example.org/a"}, {"reference_url": ""}]}
    manifest = {"documents": [{"source": "a.txt"}]}
    mapped = map_sources(records, {"a.txt": "https://example.org/a"}, manifest)
    assert mapped["records"][0]["relevant_sources"] == ["a.txt"]
    assert mapped["records"][1]["relevant_sources"] == []
    assert mapped["evidence_mapping"]["mapped"] == 1
    with pytest.raises(ValueError):
        map_sources(records, {"missing.txt": "https://example.org/a"}, manifest)
