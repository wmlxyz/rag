"""Analyze paired retrieval and QA metrics with conversation-level bootstrap."""
import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def metric(row, name):
    metrics = row.get('metrics', {})
    if name == 'answer_token_f1':
        answer = metrics.get('answer')
        return answer.get('token_f1') if answer else None
    return metrics.get({'hit_at_k': 'retrieval_hit', 'mrr_at_k': 'reciprocal_rank'}[name])


def paired_bootstrap(rows, left, right, name, seed, repetitions):
    by_id = defaultdict(dict)
    for row in rows:
        if 'error' not in row and row['strategy'] in (left, right):
            by_id[row['record_id']][row['strategy']] = row
    groups = defaultdict(list)
    for pair in by_id.values():
        if left not in pair or right not in pair:
            continue
        a, b = metric(pair[left], name), metric(pair[right], name)
        if a is None or b is None:
            continue
        groups[pair[left]['conversation_id']].append(a - b)
    if not groups:
        return {'n_pairs': 0, 'n_conversations': 0, 'difference': None, 'ci_95': None}
    keys = sorted(groups)
    observed = statistics.mean(x for values in groups.values() for x in values)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = [rng.choice(keys) for _ in keys]
        draws.append(statistics.mean(x for key in sample for x in groups[key]))
    draws.sort()
    return {'n_pairs': sum(map(len, groups.values())), 'n_conversations': len(keys), 'difference': observed,
            'ci_95': [draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--repetitions', type=int, default=5000)
    args = p.parse_args()
    if args.repetitions < 100 or Path(args.output).exists():
        raise ValueError('Use at least 100 resamples and a new output file')
    run = json.loads(Path(args.input).read_text(encoding='utf-8'))
    rows = run['results']
    pairwise = {}
    for left, right in [('history','current'), ('rewrite','current'), ('rewrite','history')]:
        pairwise[left + '_minus_' + right] = {name: paired_bootstrap(rows, left, right, name, args.seed, args.repetitions)
                                              for name in ('hit_at_k', 'mrr_at_k', 'answer_token_f1')}
    accounting = {}
    for strategy in ('current','history','rewrite'):
        group = [r for r in rows if r['strategy'] == strategy]
        good = [r for r in group if 'error' not in r]
        def usage(kind):
            entries = [r.get(kind) for r in good if r.get(kind) is not None]
            values = [u.get('total_tokens') for u in entries if u.get('total_tokens') is not None]
            return {'calls_with_usage_object': len(entries), 'calls_reporting_tokens': len(values), 'total_tokens': sum(values)}
        latency = sorted(r['timings_seconds']['total'] for r in good)
        accounting[strategy] = {'attempted': len(group), 'succeeded': len(good), 'failed': len(group)-len(good),
                                'generation_usage': usage('usage'), 'rewrite_usage': usage('rewrite_usage'),
                                'rewrite_errors': sum(bool(r.get('rewrite_error')) for r in good),
                                'invalid_citation': sum(bool(r.get('invalid_citations')) for r in good),
                                'no_valid_citation': sum(not r.get('has_valid_citation') for r in good),
                                'mean_seconds': statistics.mean(latency) if latency else None,
                                'median_seconds': statistics.median(latency) if latency else None,
                                'p95_seconds_nearest_rank': latency[math.ceil(.95*len(latency))-1] if latency else None}
    result = {'selection_note': 'Confidence intervals describe this coverage-selected candidate pool and cannot correct its selection bias.',
              'seed': args.seed, 'repetitions': args.repetitions, 'summary': run['summary'],
              'paired_differences': pairwise, 'accounting': accounting}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
