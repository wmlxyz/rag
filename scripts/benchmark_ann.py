"""Compare FAISS approximate and exact retrieval on identical query vectors."""
import argparse
import json
import statistics
import time
from pathlib import Path

from rag.cache import CachedEmbedder
from rag.cli import embedding_identity
from rag.config import load_model_config
from rag.pipeline import Store
from rag.providers.openai_compatible import OpenAICompatibleClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--index", required=True)
    p.add_argument("--db", default="data/rag.sqlite3")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=64)
    args = p.parse_args()
    if args.limit < 1 or args.top_k < 1 or Path(args.output).exists():
        raise ValueError("Use positive limits and a new output path")
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["records"]
    config = load_model_config(args.config, "embedding")
    store, embedder = Store(args.db), CachedEmbedder(OpenAICompatibleClient(config), capacity=1)
    results = []
    for record in records[:args.limit]:
        query = record["question"]
        embedder.embed([query])  # warm query embedding; timing excludes remote API
        options = dict(embedding_id=embedding_identity(config), top_k=args.top_k, min_score=-1)
        scores = {}
        for mode in (["exact", "ann"] if len(results) % 2 == 0 else ["ann", "exact"]):
            start = time.perf_counter()
            hits = store.search(args.index, query, embedder, backend=mode, ef_search=args.ef_search, **options)
            scores[mode] = hits
            scores[mode + "_seconds"] = time.perf_counter() - start
        exact_ids = {h.chunk_id for h in scores["exact"]}
        ann_ids = {h.chunk_id for h in scores["ann"]}
        results.append({"id": record["id"], "exact_seconds": scores["exact_seconds"], "ann_seconds": scores["ann_seconds"],
                        "recall_against_exact": len(exact_ids & ann_ids) / len(exact_ids) if exact_ids else None,
                        "exact_ids": [h.chunk_id for h in scores["exact"]], "ann_ids": [h.chunk_id for h in scores["ann"]]})
    if not results:
        raise ValueError("No evaluation records")
    # First ANN query includes loading and checksum validation; show it separately.
    warm = results[1:]
    output = {"index_id": args.index, "top_k": args.top_k, "ef_search": args.ef_search, "n": len(results),
              "mean_recall_against_exact": statistics.mean(r["recall_against_exact"] for r in results if r["recall_against_exact"] is not None),
              "first_ann_seconds": results[0]["ann_seconds"],
              "mean_warm_ann_seconds": statistics.mean(r["ann_seconds"] for r in warm) if warm else None,
              "mean_warm_exact_seconds": statistics.mean(r["exact_seconds"] for r in warm) if warm else None,
              "model": config.model, "cache": embedder.stats(), "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
