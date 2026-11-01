"""Compare repeated remote and warm-cache embeddings with alternating order."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from rag.cache import CachedEmbedder
from rag.config import load_model_config
from rag.providers.openai_compatible import OpenAICompatibleClient

def main():
    p = argparse.ArgumentParser()
    p.add_argument("text")
    p.add_argument("output")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--repetitions", type=int, default=5)
    args = p.parse_args()
    if args.repetitions < 1 or Path(args.output).exists():
        raise ValueError("Positive repetitions and a new output file are required")
    config = load_model_config(args.config, "embedding")
    client = OpenAICompatibleClient(config)
    cache = CachedEmbedder(client, 16)
    reference = cache.embed([args.text])[0]
    trials = []
    for i in range(args.repetitions):
        values = {}
        for mode in (["remote", "cache"] if i % 2 == 0 else ["cache", "remote"]):
            start = time.perf_counter()
            vector = (client if mode == "remote" else cache).embed([args.text])[0]
            values[mode + "_seconds"] = time.perf_counter() - start
            values[mode + "_max_abs_difference"] = max(abs(a-b) for a,b in zip(vector, reference)) if len(vector) == len(reference) else None
        trials.append(values)
    result = {"model": config.model, "base_url": config.base_url, "query_sha256": hashlib.sha256(args.text.encode()).hexdigest(),
              "warmup_calls": 1, "trials": trials, "cache_stats": cache.stats(),
              "mean_remote_seconds": statistics.mean(x["remote_seconds"] for x in trials),
              "mean_cache_seconds": statistics.mean(x["cache_seconds"] for x in trials)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
