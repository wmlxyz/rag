"""Command-line entry points for indexing and conversational retrieval."""
import argparse
import json
import logging
import os
import platform
from pathlib import Path
from datetime import datetime, timezone
import sys

from .config import load_model_config
from .errors import RagError
from .models import Message
from .providers.openai_compatible import OpenAICompatibleClient
from .pipeline import Store, answer


def embedding_identity(config):
    return f"{config.provider}:{config.base_url.rstrip('/')}:{config.model}"


def build_reranker(path, enabled):
    if not enabled:
        return None
    from .providers.dashscope_rerank import DashScopeReranker
    return DashScopeReranker(load_model_config(path, "reranker"))


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may use GBK; do not fail when a model emits emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("--config", default="configs/default.json")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat-check", help="Send one message to the configured generation service")
    chat.add_argument("message")
    embed = sub.add_parser("embed-check", help="Check the configured embedding service")
    embed.add_argument("text")
    index = sub.add_parser("index", help="Build a persistent index from a directory of UTF-8 .txt and .md files")
    index.add_argument("directory")
    index.add_argument("--db", default="data/rag.sqlite3")
    index.add_argument("--chunk-size", type=int, default=800)
    index.add_argument("--overlap", type=int, default=120)
    index.add_argument("--batch-size", type=int, default=10)
    ann = sub.add_parser("build-ann", help="Build a persistent FAISS HNSW index for an existing index")
    ann.add_argument("--db", default="data/rag.sqlite3")
    ann.add_argument("--index", required=True)
    ann.add_argument("--m", type=int, default=32)
    ann.add_argument("--ef-construction", type=int, default=100)
    lexical = sub.add_parser("build-lexical", help="Build an FTS5 index for an existing index")
    lexical.add_argument("--db", default="data/rag.sqlite3")
    lexical.add_argument("--index", required=True)
    ask = sub.add_parser("ask", help="Answer a question using a persistent index")
    ask.add_argument("question")
    ask.add_argument("--db", default="data/rag.sqlite3")
    ask.add_argument("--index", required=True)
    ask.add_argument("--session")
    ask.add_argument("--strategy", choices=("current", "history", "rewrite", "rewrite_guarded", "rewrite_fusion", "rewrite_checked", "rewrite_checked_fusion", "rewrite_verified"), default="history")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--min-score", type=float, default=0.0)
    ask.add_argument("--history-turns", type=int, default=4)
    ask.add_argument("--context-chars", type=int, default=4000)
    ask.add_argument("--context-mode", choices=("standard", "deduplicate"), default="standard")
    ask.add_argument("--citation-policy", choices=("repair", "warn"), default="repair")
    ask.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="dense")
    ask.add_argument("--backend", choices=("exact", "ann"), default="exact")
    ask.add_argument("--ef-search", type=int, default=64)
    ask.add_argument("--rerank", action="store_true")
    ask.add_argument("--candidate-k", type=int, default=20)
    ask.add_argument("--rerank-fallback", action="store_true")
    server = sub.add_parser("serve", help="Run the authenticated local HTTP API")
    server.add_argument("--db", default="data/rag.sqlite3")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--embedding-cache-size", type=int, default=0)
    server.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="dense")
    server.add_argument("--backend", choices=("exact", "ann"), default="exact")
    server.add_argument("--ef-search", type=int, default=64)
    server.add_argument("--rerank", action="store_true")
    server.add_argument("--candidate-k", type=int, default=20)
    prep = sub.add_parser("prepare-qrecc", help="Select QReCC conversations reproducibly")
    prep.add_argument("input")
    prep.add_argument("output")
    prep.add_argument("--split", choices=("train", "test"), required=True)
    prep.add_argument("--conversations", type=int, default=10)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--ground-truth", help="Optional SCAI QReCC ground-truth JSON, joined only on verified matches")
    prep.add_argument("--skip-invalid-conversations", action="store_true", help="Exclude and record conversations with inconsistent contexts")
    mapping = sub.add_parser("map-qrecc-sources", help="Map reference URLs to indexed source paths")
    mapping.add_argument("dataset")
    mapping.add_argument("mapping", help="JSON object mapping indexed paths to original URLs")
    mapping.add_argument("output")
    mapping.add_argument("--db", default="data/rag.sqlite3")
    mapping.add_argument("--index", required=True)
    subset = sub.add_parser("qrecc-subset", help="Build a bounded passage subset from a pinned stream or JSONL")
    subset.add_argument("dataset")
    subset.add_argument("output")
    input_source = subset.add_mutually_exclusive_group()
    input_source.add_argument("--jsonl", help="Local id/contents JSONL; omit to stream Hugging Face")
    input_source.add_argument("--pool", help="Previously prepared passage pool")
    subset.add_argument("--revision", default="99706cf40ae845ba9730e6a49b6ed67337bd6166")
    subset.add_argument("--max-scan", type=int, default=10000)
    subset.add_argument("--distractors", type=int, default=100)
    subset.add_argument("--max-chars", type=int, default=12000)
    subset.add_argument("--seed", type=int, default=42)
    covered = sub.add_parser("qrecc-covered-sample", help="Select conversations with matched references for a biased smoke test")
    covered.add_argument("dataset")
    covered.add_argument("output")
    covered.add_argument("--conversations", type=int, default=5)
    covered.add_argument("--seed", type=int, default=42)
    covered.add_argument("--exclude-from", help="Exclude conversations occurring in another dataset")
    followups = sub.add_parser("qrecc-followups", help="Select first labeled follow-up per conversation for diagnostics")
    followups.add_argument("dataset")
    followups.add_argument("output")
    experiment = sub.add_parser("evaluate", help="Run fixed-history baseline experiment")
    experiment.add_argument("dataset")
    experiment.add_argument("output")
    experiment.add_argument("--db", default="data/rag.sqlite3")
    experiment.add_argument("--index", required=True)
    experiment.add_argument("--strategies", nargs="+", choices=("current", "history", "rewrite", "rewrite_guarded", "rewrite_fusion", "rewrite_checked", "rewrite_checked_fusion", "rewrite_verified", "human"), default=["current", "history", "rewrite"])
    experiment.add_argument("--top-k", type=int, default=5)
    experiment.add_argument("--min-score", type=float, default=0.0)
    experiment.add_argument("--history-turns", type=int, default=4)
    experiment.add_argument("--context-chars", type=int, default=4000)
    experiment.add_argument("--context-mode", choices=("standard", "deduplicate"), default="standard")
    experiment.add_argument("--citation-policy", choices=("repair", "warn"), default="repair")
    experiment.add_argument("--embedding-cache-size", type=int, default=0)
    experiment.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="dense")
    experiment.add_argument("--backend", choices=("exact", "ann"), default="exact")
    experiment.add_argument("--ef-search", type=int, default=64)
    experiment.add_argument("--rerank", action="store_true")
    experiment.add_argument("--candidate-k", type=int, default=20)
    experiment.add_argument("--rerank-fallback", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        if args.command == "chat-check":
            client = OpenAICompatibleClient(load_model_config(args.config, "generation"))
            result = client.generate([Message(role="user", content=args.message)])
            print(result.text)
            print(f"model={result.model} total_tokens={result.usage.total_tokens}", file=sys.stderr)
        elif args.command == "embed-check":
            client = OpenAICompatibleClient(load_model_config(args.config, "embedding"))
            vector = client.embed([args.text])[0]
            print(f"dimensions={len(vector)}")
        elif args.command == "index":
            config = load_model_config(args.config, "embedding")
            def index_progress(done, total):
                if done == total or done % 100 == 0:
                    print(f"Embedding: {done}/{total} chunks", file=sys.stderr, flush=True)
            result = Store(args.db).build(args.directory, OpenAICompatibleClient(config), embedding_id=embedding_identity(config),
                                          size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size, progress=index_progress)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "build-ann":
            from .ann import build_ann
            result = build_ann(Store(args.db), args.index, m=args.m, ef_construction=args.ef_construction)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "build-lexical":
            print(json.dumps(Store(args.db).build_lexical(args.index), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            config = load_model_config(args.config, "embedding")
            result = answer(Store(args.db), OpenAICompatibleClient(config),
                            OpenAICompatibleClient(load_model_config(args.config, "generation")),
                            index_id=args.index, embedding_id=embedding_identity(config), question=args.question,
                            session_id=args.session, strategy=args.strategy, top_k=args.top_k, min_score=args.min_score,
                            history_turns=args.history_turns, context_chars=args.context_chars, context_mode=args.context_mode,
                            backend=args.backend, ef_search=args.ef_search, reranker=build_reranker(args.config, args.rerank),
                            candidate_k=args.candidate_k if args.rerank else None, rerank_fallback=args.rerank_fallback,
                            citation_policy=args.citation_policy, retrieval_mode=args.retrieval_mode)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "serve":
            from .api import serve
            from .cache import CachedEmbedder
            config = load_model_config(args.config, "embedding")
            if args.embedding_cache_size < 0:
                raise ValueError("Embedding cache size must be nonnegative")
            embedder = OpenAICompatibleClient(config)
            if args.embedding_cache_size:
                embedder = CachedEmbedder(embedder, args.embedding_cache_size)
            serve(Store(args.db), embedder,
                  OpenAICompatibleClient(load_model_config(args.config, "generation")),
                  embedding_identity(config), os.environ.get("RAG_API_TOKEN", ""), args.port,
                  backend=args.backend, ef_search=args.ef_search, reranker=build_reranker(args.config, args.rerank), candidate_k=args.candidate_k, retrieval_mode=args.retrieval_mode)
        elif args.command == "prepare-qrecc":
            from .evaluation import prepare_qrecc
            prepared = prepare_qrecc(args.input, split=args.split, conversations=args.conversations, seed=args.seed, ground_truth_path=args.ground_truth, skip_invalid_conversations=args.skip_invalid_conversations)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(prepared, handle, ensure_ascii=False, indent=2)
            print(f"Saved {len(prepared['records'])} turns to {output}")
        elif args.command == "qrecc-subset":
            from .evaluation import file_hash
            from .qrecc_corpus import HF_DATASET, build_subset, iter_hf, iter_jsonl, iter_pool
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            stream = iter_jsonl(args.jsonl) if args.jsonl else (iter_pool(args.pool) if args.pool else iter_hf(args.revision))
            source = "local JSONL" if args.jsonl else ("local passage pool" if args.pool else HF_DATASET)
            revision = file_hash(args.jsonl) if args.jsonl else (file_hash(Path(args.pool) / "manifest.json") if args.pool else args.revision)
            info = build_subset(dataset, stream, args.output, source=source, source_revision=revision,
                                seed=args.seed, max_scan=args.max_scan, distractors=args.distractors,
                                max_chars=args.max_chars, dataset_sha256=file_hash(args.dataset))
            print(json.dumps(info, ensure_ascii=False, indent=2))
        elif args.command == "qrecc-covered-sample":
            from .qrecc_corpus import covered_conversations
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            excluded = []
            if args.exclude_from:
                excluded_dataset = json.loads(Path(args.exclude_from).read_text(encoding="utf-8-sig"))
                excluded = [r["conversation_id"] for r in excluded_dataset["records"]]
            result = covered_conversations(dataset, args.conversations, args.seed, excluded)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            print(f"Saved {len(result['records'])} turns from {args.conversations} coverage-selected conversations")
        elif args.command == "qrecc-followups":
            from .qrecc_corpus import select_followups
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            selected = select_followups(dataset)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(selected, handle, ensure_ascii=False, indent=2)
            print(f"Saved {len(selected['records'])} coverage-selected follow-ups")
        elif args.command == "map-qrecc-sources":
            from .evaluation import map_sources, file_hash
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            mapping_data = json.loads(Path(args.mapping).read_text(encoding="utf-8-sig"))
            store = Store(args.db)
            with store.connect() as db:
                row = db.execute("SELECT manifest FROM indexes WHERE id=?", (args.index,)).fetchone()
            if not row:
                raise ValueError("Index version does not exist")
            mapped = map_sources(dataset, mapping_data, json.loads(row[0]))
            mapped["evidence_mapping"].update({"index_id": args.index, "mapping_sha256": file_hash(args.mapping)})
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(mapped, handle, ensure_ascii=False, indent=2)
            print(json.dumps(mapped["evidence_mapping"], ensure_ascii=False, indent=2))
        else:
            from .evaluation import file_hash, run_experiment
            output = Path(args.output)
            if output.exists():
                raise ValueError("Output already exists; choose a new path")
            dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
            mapped_index = dataset.get("evidence_mapping", {}).get("index_id")
            if mapped_index and mapped_index != args.index:
                raise ValueError("Evidence mapping was made for a different index")
            embedding_config = load_model_config(args.config, "embedding")
            generation_config = load_model_config(args.config, "generation")
            store = Store(args.db)
            with store.connect() as db:
                row = db.execute("SELECT manifest FROM indexes WHERE id=?", (args.index,)).fetchone()
            if not row:
                raise ValueError("Index version does not exist")
            if "evidence_mapping" in dataset:
                labels = {source for record in dataset["records"] for source in record.get("relevant_sources", [])}
                indexed = {d["source"] for d in json.loads(row[0])["documents"]}
                if not labels:
                    raise ValueError("No mapped reference passages; retrieval evaluation would be uninformative")
                if not labels.issubset(indexed):
                    raise ValueError("Mapped evidence sources do not match the selected index")
            options = {"top_k": args.top_k, "min_score": args.min_score, "history_turns": args.history_turns, "context_chars": args.context_chars, "context_mode": args.context_mode,
                       "backend": args.backend, "ef_search": args.ef_search,
                       "candidate_k": args.candidate_k if args.rerank else None, "rerank_fallback": args.rerank_fallback,
                       "citation_policy": args.citation_policy, "retrieval_mode": args.retrieval_mode}
            if args.embedding_cache_size < 0:
                raise ValueError("Embedding cache size must be nonnegative")
            embedder = OpenAICompatibleClient(embedding_config)
            reranker = build_reranker(args.config, args.rerank)
            if args.embedding_cache_size:
                from .cache import CachedEmbedder
                embedder = CachedEmbedder(embedder, args.embedding_cache_size)
            def progress(done, total, strategy):
                if done == 1 or done % 5 == 0 or done == total:
                    print(f"Progress: {done}/{total} ({strategy})", file=sys.stderr, flush=True)
            run = run_experiment(store, embedder, OpenAICompatibleClient(generation_config),
                                 dataset, index_id=args.index, embedding_id=embedding_identity(embedding_config), strategies=args.strategies, options=options, progress=progress, reranker=reranker)
            run["embedding_cache"] = embedder.stats() if args.embedding_cache_size else None
            run["manifest"] = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset_sha256": file_hash(args.dataset),
                "config_sha256": file_hash(args.config), "index_id": args.index, "index_manifest": json.loads(row[0]),
                "embedding_model": embedding_config.model, "embedding_base_url": embedding_config.base_url,
                "generation_model": generation_config.model, "generation_base_url": generation_config.base_url,
                "strategies": args.strategies, "options": options, "python": platform.python_version(),
                "embedding_cache_size": args.embedding_cache_size,
                "reranker_model": reranker.config.model if reranker else None,
                "history": dataset.get("history_policy", "provided reference history"), "temperature": 0.0,
                "source_sha256": {p.relative_to(Path(__file__).parent).as_posix(): file_hash(p) for p in sorted(Path(__file__).parent.rglob("*.py"))}}
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(run, handle, ensure_ascii=False, indent=2)
            print(json.dumps(run["summary"], ensure_ascii=False, indent=2))
            print(f"Results saved to {output}")
        return 0
    except (RagError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
