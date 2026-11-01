"""Paired exact/ANN and reranking diagnostics, without answer generation."""
import argparse
import json
import statistics
import time
from pathlib import Path

from rag.cache import CachedEmbedder
from rag.cli import build_reranker, embedding_identity
from rag.config import load_model_config
from rag.evaluation import file_hash
from rag.pipeline import Store
from rag.providers.openai_compatible import OpenAICompatibleClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--index", required=True)
    p.add_argument("--db", default="data/rag.sqlite3")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--ef-search", type=int, default=64)
    a = p.parse_args()
    if not 0 < a.top_k <= a.candidate_k <= 500 or a.limit < 1 or Path(a.output).exists():
        raise ValueError("Invalid limits or output exists")
    dataset = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    # Fixed ordering: first eligible follow-up per conversation, no post-hoc replacement.
    records, seen = [], set()
    for record in dataset["records"]:
        if record["conversation_id"] not in seen and record.get("history") and record.get("relevant_sources"):
            records.append(record)
            seen.add(record["conversation_id"])
            if len(records) == a.limit: break
    if not records: raise ValueError("No eligible labeled follow-ups")
    config = load_model_config(a.config, "embedding")
    emb = CachedEmbedder(OpenAICompatibleClient(config), capacity=1)
    reranker = build_reranker(a.config, True)
    store = Store(a.db)
    output = []
    for r in records:
        item = {"id": r["id"], "conversation_id": r["conversation_id"]}
        try:
            history = r["history"][-8:]
            query = "\n".join(f"{x['role']}: {x['content']}" for x in history) + "\nuser: " + r["question"]
            emb.embed([query])
            options = {"embedding_id": embedding_identity(config), "top_k": a.candidate_k, "min_score": -1, "ef_search": a.ef_search}
            rankings = {}
            for mode in (["exact", "ann"] if len(output) % 2 == 0 else ["ann", "exact"]):
                started = time.perf_counter()
                rankings[mode] = store.search(a.index, query, emb, backend=mode, **options)
                item[mode + "_seconds"] = time.perf_counter() - started
            exact, ann = rankings["exact"], rankings["ann"]
            started = time.perf_counter()
            ranked = reranker.rerank(query, [x.text for x in exact])
            item["rerank_seconds"] = time.perf_counter() - started
            item["rerank_usage"] = reranker.last_usage
            after = [exact[x.index] for x in ranked]
            relevant = set(r["relevant_sources"])
            def metrics(hits):
                ranks = [i for i, h in enumerate(hits[:a.top_k], 1) if h.source in relevant]
                return {"hit": int(bool(ranks)), "rr": 1 / min(ranks) if ranks else 0.0}
            exact_ids = {h.chunk_id for h in exact}
            item.update({"query": query, "before": metrics(exact), "after": metrics(after),
                         "candidate_recall_against_exact": len(exact_ids & {h.chunk_id for h in ann}) / len(exact_ids) if exact_ids else None,
                         "exact_top_ids": [x.chunk_id for x in exact], "reranked_top_ids": [x.chunk_id for x in after],
                         "rerank_scores": [x.score for x in ranked]})
        except Exception as exc:
            item["error"] = type(exc).__name__
        output.append(item)
        if len(output) % 5 == 0: print(f"Completed {len(output)}/{len(records)}", flush=True)
    ok = [x for x in output if "error" not in x]
    def mean(values): return statistics.mean(values) if values else None
    summary = {"attempted": len(output), "succeeded": len(ok), "failed": len(output)-len(ok),
               "before_hit_at_k": mean([x["before"]["hit"] for x in ok]), "after_hit_at_k": mean([x["after"]["hit"] for x in ok]),
               "before_mrr_at_k": mean([x["before"]["rr"] for x in ok]), "after_mrr_at_k": mean([x["after"]["rr"] for x in ok]),
               "ann_candidate_recall_against_exact": mean([x["candidate_recall_against_exact"] for x in ok if x["candidate_recall_against_exact"] is not None]),
               "mean_rerank_seconds": mean([x["rerank_seconds"] for x in ok]),
               "rerank_reported_tokens": sum((x.get("rerank_usage") or {}).get("total_tokens", 0) for x in ok),
               "rerank_usage_reported_count": sum((x.get("rerank_usage") or {}).get("total_tokens") is not None for x in ok)}
    result = {"summary": summary, "index_id": a.index, "top_k": a.top_k, "candidate_k": a.candidate_k,
              "ef_search": a.ef_search, "selection": "first labeled follow-up per conversation, fixed data order",
              "dataset_sha256": file_hash(a.dataset), "config_sha256": file_hash(a.config),
              "source_sha256": file_hash(__file__), "embedding_model": config.model, "reranker_model": reranker.config.model,
              "query": "reference history (last 8 messages) plus current question", "results": output}
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
