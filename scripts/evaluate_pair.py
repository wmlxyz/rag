"""Audit datasets and compare existing experiment outputs with clustered bootstrap."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from rag.evaluation import file_hash


def audit(dataset):
    rows = dataset["records"]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record IDs")
    return {"records": len(rows), "conversations": len({r["conversation_id"] for r in rows}),
            "with_reference_answer": sum(bool(r.get("reference_answer")) for r in rows),
            "with_reference_passages": sum(bool(r.get("reference_passages")) for r in rows),
            "with_mapped_sources": sum(bool(r.get("relevant_sources")) for r in rows),
            "mapping": dataset.get("evidence_mapping"), "selection": dataset.get("selection")}


def metric(row, name):
    m = row.get("metrics", {})
    if name == "f1": return (m.get("answer") or {}).get("token_f1")
    if name == "citation": return float(row["citation_check"]["passed"]) if row.get("evidence") and row.get("citation_check") else None
    if name == "latency": return row.get("timings_seconds", {}).get("total")
    return m.get({"hit": "retrieval_hit", "mrr": "reciprocal_rank", "context": "context_hit"}[name])


def compare(left, right, *, seed=42, samples=2000):
    def keyed(rows):
        result = {}
        for row in rows:
            key = row["record_id"]
            if key in result: raise ValueError("Duplicate record ID; select one strategy per file")
            result[key] = row
        return result
    a, b = keyed(left), keyed(right)
    result = {"left_records": len(a), "right_records": len(b), "left_only": sorted(a.keys()-b.keys()),
              "right_only": sorted(b.keys()-a.keys()), "metrics": {}}
    common = sorted(a.keys() & b.keys())
    for key in common:
        if a[key]["conversation_id"] != b[key]["conversation_id"]:
            raise ValueError("Conversation IDs differ for paired record")
    for name in ("hit", "mrr", "context", "f1", "citation", "latency"):
        groups = defaultdict(list)
        missing, failures = 0, 0
        for key in common:
            x, y = a[key], b[key]
            if "error" in x or "error" in y:
                failures += 1
                continue
            av, bv = metric(x, name), metric(y, name)
            if av is None or bv is None:
                missing += 1
                continue
            groups[x["conversation_id"]].append((key, av, bv))
        pairs = [v for group in groups.values() for v in group]
        if not pairs:
            result["metrics"][name] = {"n": 0, "missing": missing, "failures": failures}
            continue
        delta = statistics.mean(y-x for _,x,y in pairs)
        rng = random.Random(seed)
        keys = sorted(groups)
        draws = []
        if len(keys) >= 2:
            for _ in range(samples):
                chosen = [rng.choice(keys) for _ in keys]
                values = [y-x for k in chosen for _,x,y in groups[k]]
                draws.append(statistics.mean(values))
            draws.sort()
        result["metrics"][name] = {"n": len(pairs), "conversations": len(keys), "missing": missing,
            "failures": failures, "left_mean": statistics.mean(x for _,x,y in pairs),
            "right_mean": statistics.mean(y for _,x,y in pairs), "delta_right_minus_left": delta,
            "bootstrap_95_percentile": [draws[int(.025*(samples-1))], draws[int(.975*(samples-1))]] if draws else None,
            "improved": sum(y>x for _,x,y in pairs), "declined": sum(y<x for _,x,y in pairs),
            "unchanged": sum(y==x for _,x,y in pairs),
            "differences": [{"record_id": k, "left": x, "right": y} for k,x,y in pairs if x != y]}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset"); p.add_argument("left"); p.add_argument("right"); p.add_argument("output")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.bootstrap < 100: raise ValueError("At least 100 bootstrap draws required")
    def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    data, left, right = read(args.dataset), read(args.left), read(args.right)
    audit_result = audit(data)
    dataset_ids = {r["id"] for r in data["records"]}
    if any(r["record_id"] not in dataset_ids for run in (left, right) for r in run["results"]):
        raise ValueError("Result contains records absent from dataset")
    for run in (left, right):
        recorded_hash = run.get("manifest", {}).get("dataset_sha256")
        if recorded_hash and recorded_hash != file_hash(args.dataset):
            raise ValueError("Dataset hash differs from experiment manifest")
    report = {"inputs": {"dataset_sha256": file_hash(args.dataset), "left_sha256": file_hash(args.left),
               "right_sha256": file_hash(args.right), "script_sha256": file_hash(__file__),
               "seed": args.seed, "bootstrap_draws": args.bootstrap}, "dataset_audit": audit_result,
              "left_manifest": left.get("manifest"), "right_manifest": right.get("manifest"),
              "comparison": compare(left["results"], right["results"], seed=args.seed, samples=args.bootstrap),
              "limitations": ["Pairs with errors or missing metrics are excluded separately per metric", "Bootstrap resamples conversations; intervals do not remove selection bias", "Source match and citation syntax do not establish evidence support", "Check manifests for incompatible experimental settings"]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["comparison"]["metrics"], ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
