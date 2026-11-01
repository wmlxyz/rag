import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("paired", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pair.py")
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)

def row(i, score, conv="c"):
    return {"record_id": i, "conversation_id": conv, "metrics": {"retrieval_hit": score}}

def test_pair_missing_failures_and_cluster_bootstrap():
    a = [row("1", 0, "a"), row("2", 1, "b"), row("3", None, "b")]
    b = [row("1", 1, "a"), row("2", 0, "b"), row("3", 1, "b")]
    m = paired.compare(a, b, samples=200)["metrics"]["hit"]
    assert m["n"] == 2 and m["missing"] == 1 and m["delta_right_minus_left"] == 0
    assert m["improved"] == 1 and m["declined"] == 1
    assert m["bootstrap_95_percentile"] == [-1, 1]
    b[1] = {"record_id": "2", "conversation_id": "b", "error": "ProviderError"}
    m = paired.compare(a, b, samples=200)["metrics"]["hit"]
    assert m["failures"] == 1 and m["n"] == 1 and m["bootstrap_95_percentile"] is None

def test_duplicate_or_inconsistent_pair_rejected():
    with pytest.raises(ValueError): paired.compare([row("1", 1), row("1", 0)], [])
    with pytest.raises(ValueError): paired.compare([row("1", 1, "a")], [row("1", 1, "b")])
