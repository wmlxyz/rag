"""Select one annotated follow-up per randomly sampled conversation."""
import argparse
import json
import random
from pathlib import Path
from rag.evaluation import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    source = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
    excluded = set()
    for path in args.exclude:
        other = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        excluded.update(r["conversation_id"] for r in other["records"])
    eligible = {}
    for row in source["records"]:
        conv = row["conversation_id"]
        if conv not in excluded and row.get("history") and row.get("relevant_sources") and row.get("reference_answer"):
            eligible.setdefault(conv, row)
    if len(eligible) < args.count:
        raise ValueError(f"Only {len(eligible)} eligible conversations; requested {args.count}")
    chosen = sorted(random.Random(args.seed).sample(sorted(eligible), args.count))
    result = dict(source)
    result["records"] = [eligible[conv] for conv in chosen]
    result["selection"] = {"method": "random sample of eligible conversation IDs; first labeled follow-up per conversation",
                           "seed": args.seed, "count": args.count, "eligible_conversations": len(eligible),
                           "source_sha256": file_hash(args.dataset), "excluded_sha256": [file_hash(p) for p in args.exclude],
                           "limitation": "selected for evidence coverage; source candidate pool uses test passage labels and is not representative"}
    result["evidence_mapping"] = {"method": "exact annotated passage ID", "total_records": args.count,
                                  "records_with_matched_passage": args.count, "limitation": "coverage-selected"}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Selected {len(chosen)} of {len(eligible)} eligible conversations")


if __name__ == "__main__":
    main()
