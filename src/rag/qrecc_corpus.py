"""Build a bounded, explicitly restricted QReCC passage candidate pool."""
import hashlib
import json
import itertools
import random
import shutil
import tempfile
from pathlib import Path

from .errors import RagError
HF_DATASET = "slupart/qrecc-passages"
HF_REVISION = "99706cf40ae845ba9730e6a49b6ed67337bd6166"


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_pool(path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["passages"]:
        filename = item["source"]
        if Path(filename).name != filename:
            raise ValueError("Invalid pool document path")
        content = (root / "documents" / filename).read_text(encoding="utf-8")
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != item["sha256"]:
            raise ValueError("Pool document hash mismatch")
        yield {"id": item["id"], "contents": content}


def covered_conversations(dataset, count, seed, excluded=None):
    if count < 1:
        raise ValueError("Conversation count must be positive")
    candidates = sorted({r["conversation_id"] for r in dataset["records"] if r.get("relevant_sources")} - set(excluded or ()))
    if len(candidates) < count:
        raise ValueError("Not enough conversations with matched passages")
    chosen = set(random.Random(seed).sample(candidates, count))
    result = dict(dataset)
    result["records"] = [r for r in dataset["records"] if r["conversation_id"] in chosen]
    result["evidence_mapping"] = {"method": "inherited exact annotated passage ID", "total_records": len(result["records"]),
                                  "records_with_matched_passage": sum(bool(r.get("relevant_sources")) for r in result["records"]),
                                  "limitation": "coverage-selected sample; not representative"}
    result["selection"] = "coverage-selected conversations; entire conversations retained; biased smoke test, not a random test sample"
    result["selection_seed"] = seed
    result["selected_conversations"] = sorted(chosen)
    return result


def select_followups(dataset):
    """For small development diagnostics, keep first labeled follow-up per conversation."""
    selected, seen = [], set()
    for record in dataset["records"]:
        conv = record["conversation_id"]
        if conv not in seen and record.get("history") and record.get("relevant_sources") and record.get("reference_answer"):
            selected.append(record)
            seen.add(conv)
    if not selected:
        raise ValueError("No labeled follow-ups found")
    result = dict(dataset)
    result["records"] = selected
    result["selection"] = "first matched follow-up with an answer per covered conversation; biased diagnostic only"
    result["evidence_mapping"] = {"method": "exact annotated passage ID", "total_records": len(selected),
                                  "records_with_matched_passage": len(selected), "limitation": "coverage-selected diagnostic only"}
    return result


def iter_hf(revision=HF_REVISION):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError("Install the optional dependency: pip install -e '.[qrecc]'") from exc
    try:
        yield from load_dataset(HF_DATASET, revision=revision, split="train", streaming=True)
    except Exception as exc:
        raise RagError(f"QReCC stream failed: {type(exc).__name__}") from None


def build_subset(dataset, stream, output, *, source, source_revision, seed=42,
                 max_scan=10000, distractors=100, max_chars=12000, dataset_sha256=None):
    if max_scan < 1 or distractors < 1 or max_chars < 1:
        raise ValueError("max_scan, distractors and max_chars must be positive")
    if any("reference_passages" not in r for r in dataset["records"]):
        raise ValueError("Dataset lacks reference_passages; regenerate with prepare-qrecc")
    targets = {p for r in dataset["records"] for p in r["reference_passages"] if isinstance(p, str) and p}
    if not targets:
        raise ValueError("No annotated passage IDs in dataset")
    rng = random.Random(seed)
    relevant, reservoir = {}, []
    eligible, scanned, empty, oversized, malformed, duplicate = 0, 0, 0, 0, 0, 0
    seen = set()
    # The bounded scan is intentional: reference passages outside the scanned prefix remain missing.
    for row in itertools.islice(stream, max_scan):
        scanned += 1
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("contents"), str) or not row["id"]:
            malformed += 1
            continue
        passage_id, content = row["id"], row["contents"]
        if passage_id in seen:
            duplicate += 1
            continue
        seen.add(passage_id)
        if not content.strip():
            empty += 1
            continue
        if len(content) > max_chars:
            oversized += 1
            continue
        if passage_id in targets:
            relevant[passage_id] = content
        else:
            eligible += 1
            if len(reservoir) < distractors:
                reservoir.append((passage_id, content))
            else:
                j = rng.randrange(eligible)
                if j < distractors:
                    reservoir[j] = (passage_id, content)
    selected = sorted(list(relevant.items()) + reservoir)
    if not selected:
        raise ValueError("No usable passages found; no output published")
    output = Path(output)
    if output.exists():
        raise ValueError("Output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".qrecc-subset-", dir=output.parent))
    try:
        documents = staging / "documents"
        documents.mkdir()
        names = {}
        passages = []
        for passage_id, content in selected:
            filename = hashlib.sha256(passage_id.encode("utf-8")).hexdigest() + ".txt"
            (documents / filename).write_text(content, encoding="utf-8", newline="")
            names[passage_id] = filename
            passages.append({"id": passage_id, "source": filename, "sha256": hashlib.sha256(content.encode('utf-8')).hexdigest(), "reference": passage_id in targets})
        mapped = dict(dataset)
        mapped["records"] = []
        for record in dataset["records"]:
            item = dict(record)
            item["relevant_sources"] = sorted({names[p] for p in record["reference_passages"] if p in relevant})
            mapped["records"].append(item)
        coverage = {"total_records": len(mapped["records"]), "records_with_reference_ids": sum(bool(r["reference_passages"]) for r in mapped["records"]),
                    "records_with_matched_passage": sum(bool(r["relevant_sources"]) for r in mapped["records"]),
                    "reference_ids": len(targets), "reference_ids_found": len(relevant), "missing_reference_ids": sorted(targets - relevant.keys())}
        mapped["evidence_mapping"] = {"method": "exact annotated passage ID", **coverage,
                                      "limitation": "missing IDs excluded from retrieval metric denominator; candidate pool selected using test labels"}
        manifest = {"schema_version": 1, "source": source, "revision": source_revision, "dataset_sha256": dataset_sha256,
                    "selection": "all matching annotated IDs in scanned prefix plus reservoir-sampled non-target distractors",
                    "seed": seed, "max_scan": max_scan, "rows_scanned": scanned, "max_chars": max_chars,
                    "eligible_distractors": eligible, "distractors_selected": len(reservoir), "empty": empty,
                    "oversized": oversized, "malformed": malformed, "duplicate_ids": duplicate,
                    "coverage": coverage, "passages": passages}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "mapped_dataset.json").write_text(json.dumps(mapped, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(output)
    finally:
        if staging.exists() and not staging.is_symlink() and staging.resolve().parent == output.parent.resolve():
            shutil.rmtree(staging)
    return {"output": str(output), "passages_selected": len(selected), **coverage, "rows_scanned": scanned}
