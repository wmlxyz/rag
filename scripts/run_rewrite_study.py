"""Run a fixed paired Query Rewrite study with durable per-record checkpoints."""
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from rag.cli import embedding_identity, build_reranker
from rag.config import load_model_config
from rag.evaluation import file_hash, run_record, summarize
from rag.pipeline import Store
from rag.providers.openai_compatible import OpenAICompatibleClient


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'src' / 'rag').rglob('*.py')) + [Path(__file__).resolve()]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('output')
    p.add_argument('--index', required=True)
    p.add_argument('--db', default='data/rag.sqlite3')
    p.add_argument('--config', default='configs/default.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--backend', choices=('exact', 'ann'), default='ann')
    p.add_argument('--ef-search', type=int, default=64)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--strategies', nargs='+', default=['current', 'history', 'rewrite'], choices=['current', 'history', 'rewrite', 'rewrite_guarded', 'rewrite_fusion', 'rewrite_checked', 'rewrite_checked_fusion', 'rewrite_verified', 'human'])
    p.add_argument('--retrieval-mode', choices=['dense', 'hybrid'], default='dense')
    p.add_argument('--rerank', action='store_true')
    p.add_argument('--candidate-k', type=int, default=20)
    args = p.parse_args()
    output = Path(args.output)
    checkpoint = Path(str(output) + '.jsonl')
    if output.exists():
        raise ValueError('Final output exists; select a new output path')
    dataset = json.loads(Path(args.dataset).read_text(encoding='utf-8-sig'))
    records = dataset['records']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Dataset must contain unique record IDs')
    emb_cfg = load_model_config(args.config, 'embedding')
    gen_cfg = load_model_config(args.config, 'generation')
    store = Store(args.db)
    with store.connect() as db:
        index_row = db.execute('SELECT manifest FROM indexes WHERE id=?', (args.index,)).fetchone()
        if not index_row:
            raise ValueError('Index does not exist')
        if args.backend == 'ann':
            ann_row = db.execute('SELECT revision, sha256 FROM ann_revisions WHERE index_id=?', (args.index,)).fetchone()
            if not ann_row:
                raise ValueError('ANN index does not exist')
        else:
            ann_row = None
    index_manifest = json.loads(index_row[0])
    indexed_sources = {item['source'] for item in index_manifest['documents']}
    if any(not set(r.get('relevant_sources', [])).issubset(indexed_sources) for r in records):
        raise ValueError('Reference sources do not match selected index')
    options = {'top_k': args.top_k, 'min_score': 0.0, 'history_turns': 4, 'context_chars': 4000,
               'context_mode': 'standard', 'backend': args.backend, 'ef_search': args.ef_search,
               'citation_policy': 'repair', 'retrieval_mode': args.retrieval_mode,
               'candidate_k': args.candidate_k if args.rerank else None}
    strategies = args.strategies
    if len(set(strategies)) != len(strategies):
        raise ValueError('Duplicate strategies')
    reranker = build_reranker(args.config, args.rerank)
    identity = {'dataset_sha256': file_hash(args.dataset), 'config_sha256': file_hash(args.config),
                'index_id': args.index, 'index_manifest_sha256': hashlib.sha256(index_row[0].encode('utf-8')).hexdigest(),
                'ann_revision': ann_row[0] if ann_row else None, 'ann_sha256': ann_row[1] if ann_row else None,
                'embedding_model': emb_cfg.model, 'embedding_base_url': emb_cfg.base_url,
                'generation_model': gen_cfg.model, 'generation_base_url': gen_cfg.base_url,
                'reranker_model': reranker.config.model if reranker else None,
                'reranker_base_url': reranker.config.base_url if reranker else None,
                'options': options, 'strategies': strategies, 'source_sha256': source_hashes(),
                'python': platform.python_version(), 'history': dataset.get('history_policy'), 'temperature': 0.0}
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if checkpoint.exists():
        if not args.resume:
            raise ValueError('Checkpoint exists; use --resume to continue')
        with checkpoint.open(encoding='utf-8') as f:
            header = json.loads(next(f))
            if header.get('identity') != identity:
                raise ValueError('Checkpoint configuration, source code or index differs; refusing resume')
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        if args.resume:
            raise ValueError('Checkpoint does not exist')
        with checkpoint.open('x', encoding='utf-8') as f:
            f.write(json.dumps({'identity': identity, 'started_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + '\n')
            f.flush(); os.fsync(f.fileno())
    expected = {(s, r['id']) for s in strategies for r in records}
    completed = {(r['strategy'], r['record_id']) for r in rows}
    if len(completed) != len(rows) or not completed.issubset(expected):
        raise ValueError('Checkpoint contains duplicate or unexpected records')
    emb, gen = OpenAICompatibleClient(emb_cfg), OpenAICompatibleClient(gen_cfg)
    with checkpoint.open('a', encoding='utf-8') as f:
        for strategy in strategies:
            for record in records:
                if (strategy, record['id']) in completed:
                    continue
                try:
                    result = run_record(store, emb, gen, record, index_id=args.index,
                                        embedding_id=embedding_identity(emb_cfg), strategy=strategy, options=options, reranker=reranker)
                except Exception as exc:
                    result = {'record_id': record['id'], 'conversation_id': record['conversation_id'],
                              'strategy': strategy, 'error': type(exc).__name__}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush(); os.fsync(f.fileno())
                rows.append(result)
                if len(rows) == 1 or len(rows) % 5 == 0 or len(rows) == len(expected):
                    print(f'Progress: {len(rows)}/{len(expected)} ({strategy})', flush=True)
    result = {'manifest': identity, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'source_unchanged': source_hashes() == identity['source_sha256'], 'summary': summarize(rows), 'results': rows}
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
