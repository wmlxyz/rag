"""Calibrate a retrieval threshold from explicitly labelled examples."""
import argparse
import json
from pathlib import Path
from rag.cli import embedding_identity
from rag.config import load_model_config
from rag.evaluation import file_hash
from rag.pipeline import Store
from rag.providers.openai_compatible import OpenAICompatibleClient


def metrics(rows, threshold):
    tp = sum(r['score'] >= threshold and r['answerable'] for r in rows)
    fn = sum(r['score'] < threshold and r['answerable'] for r in rows)
    fp = sum(r['score'] >= threshold and not r['answerable'] for r in rows)
    tn = sum(r['score'] < threshold and not r['answerable'] for r in rows)
    return {'n': len(rows), 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'balanced_accuracy': ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if tp + fn and tn + fp else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset'); p.add_argument('output'); p.add_argument('--index', required=True)
    p.add_argument('--db', default='data/rag.sqlite3'); p.add_argument('--config', default='configs/default.json')
    args = p.parse_args()
    if Path(args.output).exists(): raise ValueError('Output already exists')
    rows = json.loads(Path(args.dataset).read_text(encoding='utf-8'))['records']
    if any(type(r.get('answerable')) is not bool or r.get('split') not in ('dev', 'validation') for r in rows):
        raise ValueError('Records require boolean answerable and dev/validation split')
    config = load_model_config(args.config, 'embedding')
    store, embedder = Store(args.db), OpenAICompatibleClient(config)
    scored = []
    for row in rows:
        hits = store.search(args.index, row['question'], embedder, embedding_id=embedding_identity(config), top_k=1, min_score=-1)
        if not hits: raise ValueError('Index returned no result')
        scored.append({**row, 'score': hits[0].score, 'source': hits[0].source})
    dev = [r for r in scored if r['split'] == 'dev']
    val = [r for r in scored if r['split'] == 'validation']
    thresholds = sorted({-1.0, 1.0, *(r['score'] for r in dev)})
    if not dev or not val or not any(r['answerable'] for r in dev) or not any(not r['answerable'] for r in dev):
        raise ValueError('Development set requires positive and negative examples; validation is required')
    threshold = max(thresholds, key=lambda t: (metrics(dev, t)['balanced_accuracy'], t))
    result = {'index_id': args.index, 'model': config.model, 'dataset_sha256': file_hash(args.dataset),
              'threshold': threshold, 'criterion': 'maximum development balanced accuracy; higher threshold on ties',
              'dev': metrics(dev, threshold), 'validation': metrics(val, threshold), 'records': scored,
              'limitation': 'small hand-labelled single-knowledge-base example; similarity is not proof of answer support'}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
