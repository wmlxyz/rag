import json
import pytest
from rag.qrecc_corpus import build_subset, covered_conversations, iter_pool, select_followups


def test_selection_mapping_and_determinism(tmp_path):
    dataset = {"records": [{"id": "1", "reference_passages": ["p1", "p4"]}, {"id": "2", "reference_passages": ["missing"]}, {"id": "3", "reference_passages": []}]}
    rows = [{"id": f"p{i}", "contents": f"text {i}"} for i in range(1, 10)]
    first = build_subset(dataset, iter(rows), tmp_path / "a", source="test", source_revision="v1", seed=1, max_scan=6, distractors=2)
    second = build_subset(dataset, iter(rows), tmp_path / "b", source="test", source_revision="v1", seed=1, max_scan=6, distractors=2)
    assert first["reference_ids_found"] == 2
    assert first["records_with_matched_passage"] == 1
    assert first["missing_reference_ids"] == ["missing"]
    ma = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
    assert ma == mb and len(ma["passages"]) == 4
    mapped = json.loads((tmp_path / "a" / "mapped_dataset.json").read_text(encoding="utf-8"))
    assert len(mapped["records"][0]["relevant_sources"]) == 2
    assert mapped["records"][1]["relevant_sources"] == []
    assert all((tmp_path / "a" / "documents" / p["source"]).exists() for p in ma["passages"])
    assert len(list(iter_pool(tmp_path / "a"))) == 4


def test_missing_reference_field_and_failed_stream_no_output(tmp_path):
    with pytest.raises(ValueError, match="reference_passages"):
        build_subset({"records": [{"id": "1"}]}, [], tmp_path / "missing", source="x", source_revision="x")
    def broken():
        yield {"id": "p1", "contents": "yes"}
        raise RuntimeError("stream broke")
    with pytest.raises(RuntimeError):
        build_subset({"records": [{"reference_passages": ["p1"]}]}, broken(), tmp_path / "failed", source="x", source_revision="x")
    assert not (tmp_path / "failed").exists()


def test_select_covered_preserves_conversation():
    dataset = {"records": [{"conversation_id": "a", "relevant_sources": ["a.txt"]},
                            {"conversation_id": "a", "relevant_sources": []},
                            {"conversation_id": "b", "relevant_sources": []}]}
    selected = covered_conversations(dataset, 1, 42)
    assert len(selected["records"]) == 2
    assert selected["selected_conversations"] == ["a"]
    assert "biased" in selected["selection"]
    with pytest.raises(ValueError):
        covered_conversations(dataset, 1, 42, excluded={"a"})


def test_followups_are_one_per_conversation():
    dataset = {"records": [
        {"conversation_id": "a", "history": [], "relevant_sources": ["p"], "reference_answer": "x"},
        {"conversation_id": "a", "history": [1], "relevant_sources": ["p"], "reference_answer": "x"},
        {"conversation_id": "a", "history": [1], "relevant_sources": ["q"], "reference_answer": "y"},
        {"conversation_id": "b", "history": [1], "relevant_sources": [], "reference_answer": "z"},
    ]}
    out = select_followups(dataset)
    assert len(out["records"]) == 1 and out["records"][0]["relevant_sources"] == ["p"]
