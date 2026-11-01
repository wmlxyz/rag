"""Reproducible fixed-history baseline runs and transparent basic metrics."""
import collections
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

from .models import Message
from .pipeline import answer


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_qrecc(path, *, split, conversations, seed, ground_truth_path=None, skip_invalid_conversations=False):
    if split not in ("train", "test") or conversations < 1:
        raise ValueError("Invalid split or conversation count")
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected QReCC JSON array")
    grouped = collections.defaultdict(list)
    ground_truth = {}
    if ground_truth_path:
        for record in json.loads(Path(ground_truth_path).read_text(encoding="utf-8")):
            key = (record["Conversation_no"], record["Turn_no"])
            if key in ground_truth:
                raise ValueError("Duplicate ground-truth conversation/turn ID")
            ground_truth[key] = record
    for row in rows:
        grouped[str(row["Conversation_no"])].append(row)
    ids = sorted(grouped)
    if conversations > len(ids):
        raise ValueError("Requested more conversations than available")
    chosen = sorted(random.Random(seed).sample(ids, conversations))
    output, invalid = [], []
    for conv in chosen:
        history = []
        conversation_records = []
        for row in sorted(grouped[conv], key=lambda x: x["Turn_no"]):
            context = row["Context"]
            if not isinstance(context, list) or context != [item["content"] for item in history]:
                if skip_invalid_conversations:
                    invalid.append(conv)
                    conversation_records = []
                    break
                raise ValueError("QReCC context does not match preceding rewritten questions and nonempty answers")
            gt = ground_truth.get((row["Conversation_no"], row["Turn_no"])) if ground_truth_path else None
            gt_verified = bool(gt and gt.get("Truth_rewrite") == row.get("Rewrite") and gt.get("Truth_answer") == row.get("Answer"))
            passages = (gt.get("Truth_passages") or []) if gt_verified else (row.get("Passages") or [])
            conversation_records.append({"id": f"{split}:{conv}:{row['Turn_no']}", "conversation_id": f"{split}:{conv}",
                           "question": row["Question"], "history": list(history), "human_rewrite": row.get("Rewrite"),
                           "reference_answer": row.get("Answer"), "reference_url": row.get("Answer_URL"),
                           "reference_passages": passages, "supplemental_truth_verified": gt_verified if ground_truth_path else None,
                           "conversation_source": row.get("Conversation_source"), "relevant_sources": []})
            history.append({"role": "user", "content": row["Rewrite"]})
            if row.get("Answer"):
                history.append({"role": "assistant", "content": row["Answer"]})
        output.extend(conversation_records)
    return {"dataset": "QReCC", "split": split, "seed": seed, "source_sha256": file_hash(path),
            "ground_truth_sha256": file_hash(ground_truth_path) if ground_truth_path else None,
            "ground_truth_policy": "attach only when conversation ID, turn ID, rewrite, and answer match exactly" if ground_truth_path else None,
            "excluded_context_mismatch_conversations": invalid,
            "source": "https://zenodo.org/records/5760304", "license": "CC BY-SA 3.0",
            "selection": "Uniform sample of conversation IDs; all turns retained",
            "history_policy": "Official Context: previous human rewrites and nonempty reference answers", "records": output}


def tokens(text):
    text = re.sub(r"\[\d+\]", "", text.lower())
    result, word = [], []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            if word:
                result.append(''.join(word)); word = []
            result.append(char)
        elif char.isalnum():
            word.append(char)
        else:
            if word:
                result.append(''.join(word)); word = []
    if word:
        result.append(''.join(word))
    return result


def answer_scores(prediction, reference):
    if not reference or not reference.strip():
        return None
    pred, gold = tokens(prediction), tokens(reference)
    if not gold:
        return None
    overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    f1 = 2 * overlap / (len(pred) + len(gold)) if pred else 0.0
    return {"exact_match": float(pred == gold), "token_f1": f1}


def map_sources(dataset, mapping, manifest):
    """Map exact reference URLs to indexed source paths; unmatched items remain unlabeled."""
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not (isinstance(v, str) and v or isinstance(v, list) and v and all(isinstance(u, str) and u for u in v)) for k, v in mapping.items()):
        raise ValueError("Mapping must map indexed source paths to a URL or a nonempty URL list")
    indexed = {doc["source"] for doc in manifest["documents"]}
    if set(mapping) - indexed:
        raise ValueError("Source mapping includes files not present in the index")
    result = dict(dataset)
    result["records"] = []
    for record in dataset["records"]:
        item = dict(record)
        item["relevant_sources"] = sorted(source for source, url in mapping.items() if item.get("reference_url") in ([url] if isinstance(url, str) else url))
        result["records"].append(item)
    result["evidence_mapping"] = {"method": "exact Answer_URL to index source path", "mapped": sum(bool(r["relevant_sources"]) for r in result["records"]),
                                  "total": len(result["records"]), "limitation": "document-level URL match; unmapped records excluded from retrieval metrics"}
    return result


def run_record(store, embedder, generator, record, *, index_id, embedding_id, strategy, options, reranker=None):
    history = [Message(item["role"], item["content"]) for item in record["history"]]
    override = record.get("human_rewrite") if strategy == "human" else None
    if strategy == "human" and not override:
        raise ValueError("Human rewrite is missing")
    result = answer(store, embedder, generator, index_id=index_id, embedding_id=embedding_id,
                    question=record["question"], session_id="evaluation", strategy="current" if strategy == "human" else strategy,
                    history_override=history, query_override=override, persist=False, reranker=reranker, **options)
    relevant = set(record.get("relevant_sources") or [])
    ranks = [i for i, hit in enumerate(result["retrieved"], 1) if hit["source"] in relevant]
    result["metrics"] = {"answer": answer_scores(result["answer"], record.get("reference_answer")),
                         "retrieval_hit": float(bool(ranks)) if relevant else None,
                         "context_hit": float(any(e["source"] in relevant for e in result["evidence"])) if relevant else None,
                         "reciprocal_rank": 1 / min(ranks) if ranks else (0.0 if relevant else None)}
    result["record_id"] = record["id"]
    result["conversation_id"] = record["conversation_id"]
    result["strategy"] = strategy
    return result


def summarize(rows):
    summary = {}
    for strategy in sorted({r["strategy"] for r in rows}):
        group = [r for r in rows if r["strategy"] == strategy]
        good = [r for r in group if "error" not in r]
        def mean(values):
            vals = [v for v in values if v is not None]
            return {"mean": statistics.mean(vals) if vals else None, "n": len(vals)}
        summary[strategy] = {"attempted": len(group), "succeeded": len(good), "failed": len(group) - len(good),
            "answer_token_f1": mean([r["metrics"]["answer"]["token_f1"] if r["metrics"]["answer"] else None for r in good]),
            "answer_exact_match": mean([r["metrics"]["answer"]["exact_match"] if r["metrics"]["answer"] else None for r in good]),
            "document_hit_at_k": mean([r["metrics"]["retrieval_hit"] for r in good]),
            "document_mrr_at_k": mean([r["metrics"]["reciprocal_rank"] for r in good]),
            "context_source_hit": mean([r["metrics"]["context_hit"] for r in good]),
            "context_chars": mean([r["context_chars_used"] for r in good]),
            "valid_citation_fraction": mean([float(r.get("citation_check", {}).get("passed", r["has_valid_citation"] and not r["invalid_citations"])) for r in good if r["evidence"]]),
            "total_latency_seconds": mean([r["timings_seconds"]["total"] for r in good])}
    return summary


def run_experiment(store, embedder, generator, dataset, *, index_id, embedding_id, strategies, options, progress=None, reranker=None):
    records = dataset["records"]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must contain unique record IDs")
    rows = []
    for strategy in strategies:
        for record in records:
            try:
                result = run_record(store, embedder, generator, record, index_id=index_id,
                                    embedding_id=embedding_id, strategy=strategy, options=options, reranker=reranker)
            except Exception as exc:
                # Never record provider response bodies, credentials, or exception messages.
                result = {"record_id": record["id"], "conversation_id": record["conversation_id"],
                          "strategy": strategy, "error": type(exc).__name__}
            rows.append(result)
            if progress:
                progress(len(rows), len(records) * len(strategies), strategy)
    return {"results": rows, "summary": summarize(rows)}
