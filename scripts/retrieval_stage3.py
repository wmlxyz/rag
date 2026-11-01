"""Paired retrieval study: one query/ANN recall per example, multiple rerank settings."""
import argparse
import json
import statistics
import time
from pathlib import Path
from rag.cli import build_reranker, embedding_identity
from rag.config import load_model_config
from rag.evaluation import file_hash
from rag.pipeline import Store, reciprocal_rank_fusion
from rag.providers.openai_compatible import OpenAICompatibleClient


def metrics(hits, relevant, k):
    ranks = [i for i, h in enumerate(hits[:k], 1) if h.source in relevant]
    return {"hit": int(bool(ranks)), "rr": 1 / ranks[0] if ranks else 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--index", required=True)
    p.add_argument("--db", default="data/rag.sqlite3")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--limit", type=int, default=12)
    a = p.parse_args()
    path = Path(a.output)
    if path.exists() or a.limit < 1:
        raise ValueError("Output already exists or invalid limit")
    data = json.loads(Path(a.dataset).read_text(encoding="utf-8-sig"))
    if data.get("evidence_mapping", {}).get("index_id") not in (None, a.index):
        raise ValueError("Dataset evidence mapping does not match index")
    records = [r for r in data["records"] if r.get("history") and r.get("relevant_sources")][:a.limit]
    if not records:
        raise ValueError("No labeled follow-ups")
    embcfg = load_model_config(a.config, "embedding")
    emb = OpenAICompatibleClient(embcfg)
    reranker = build_reranker(a.config, True)
    store = Store(a.db)
    result = {"manifest": {"dataset_sha256": file_hash(a.dataset), "config_sha256": file_hash(a.config),
        "source_sha256": {str(p): file_hash(p) for p in [Path(__file__), *sorted(Path("src/rag").rglob("*.py"))]},
        "index_id": a.index, "embedding_model": embcfg.model, "reranker_model": reranker.config.model,
        "selection": "first eligible labeled follow-ups in dataset order", "query": "last 8 reference history messages plus current question",
        "backend": "ann", "ef_search": 64, "top_k": 5, "candidate_k": [10, 20], "min_score": 0.0}, "results": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation and write after each complete example preserves partial results on interruption.
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    def save():
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    for record in records:
        item = {"id": record["id"], "conversation_id": record["conversation_id"], "variants": {}}
        try:
            query = "\n".join(f"{x['role']}: {x['content']}" for x in record["history"][-8:]) + "\nuser: " + record["question"]
            item["query"] = query
            item["relevant_sources"] = record["relevant_sources"]
            t = time.perf_counter()
            dense = store.search(a.index, query, emb, embedding_id=embedding_identity(embcfg), top_k=20, min_score=0, backend="ann")
            item["dense_seconds_including_embedding"] = time.perf_counter() - t
            t = time.perf_counter()
            lexical = store.search_lexical(a.index, query, top_k=20)
            item["lexical_seconds"] = time.perf_counter() - t
            for k in (10, 20):
                for mode in ("dense", "hybrid"):
                    candidates = dense[:k] if mode == "dense" else reciprocal_rank_fusion([dense[:k], lexical[:k]], k)
                    key = f"{mode}_{k}"
                    v = {"candidate_metrics": metrics(candidates, set(record["relevant_sources"]), k),
                         "before": metrics(candidates, set(record["relevant_sources"]), 5),
                         "candidates": [{"chunk_id": h.chunk_id, "source": h.source} for h in candidates]}
                    item["variants"][key] = v
                    if not candidates:
                        v["error"] = "no_candidates"
                        continue
                    try:
                        t = time.perf_counter()
                        ranked = reranker.rerank(query, [h.text for h in candidates])
                        v["rerank_seconds"] = time.perf_counter() - t
                        v["rerank_usage"] = reranker.last_usage
                        if len(ranked) != len(candidates) or {x.index for x in ranked} != set(range(len(candidates))):
                            raise ValueError("invalid reranking response")
                        after = [candidates[x.index] for x in ranked]
                        v["after"] = metrics(after, set(record["relevant_sources"]), 5)
                        v["reranked"] = [{"chunk_id": h.chunk_id, "source": h.source} for h in after]
                    except Exception as exc:
                        v["error"] = type(exc).__name__
        except Exception as exc:
            item["error"] = type(exc).__name__
        result["results"].append(item)
        save()
        print(f"Completed {len(result['results'])}/{len(records)}", flush=True)
    summary = {}
    for mode in ("dense", "hybrid"):
        for k in (10, 20):
            key = f"{mode}_{k}"
            rows = [r["variants"][key] for r in result["results"] if key in r["variants"] and "after" in r["variants"][key]]
            mean = lambda values: statistics.mean(values) if values else None
            summary[key] = {"success": len(rows), "failed": len(records)-len(rows),
                "candidate_hit": mean([r["candidate_metrics"]["hit"] for r in rows]),
                "before_hit5": mean([r["before"]["hit"] for r in rows]), "before_mrr5": mean([r["before"]["rr"] for r in rows]),
                "after_hit5": mean([r["after"]["hit"] for r in rows]), "after_mrr5": mean([r["after"]["rr"] for r in rows]),
                "mean_rerank_seconds": mean([r["rerank_seconds"] for r in rows]),
                "rerank_reported_tokens": sum((r.get("rerank_usage") or {}).get("total_tokens", 0) or 0 for r in rows),
                "usage_reported": sum((r.get("rerank_usage") or {}).get("total_tokens") is not None for r in rows)}
    result["summary"] = summary
    save()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
