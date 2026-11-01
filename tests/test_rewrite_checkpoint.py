import importlib.util
import json
import sys
from pathlib import Path
import pytest
from rag.config import ModelConfig
from rag.pipeline import Store


def load_runner():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_rewrite_study.py'
    spec = importlib.util.spec_from_file_location('rewrite_study_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_resume_without_duplicate_calls(tmp_path, monkeypatch):
    module = load_runner()
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'a.txt').write_text('alpha', encoding='utf-8')
    class Emb:
        def embed(self, texts): return [[1.0] for _ in texts]
    store = Store(tmp_path / 'db.sqlite')
    idx = store.build(docs, Emb(), embedding_id='openai_compatible:https://example.test:m', size=50, overlap=0)['index_id']
    records = [{'id': str(i), 'conversation_id': str(i), 'question': 'q', 'history': [], 'relevant_sources': ['a.txt']} for i in range(2)]
    dataset = tmp_path / 'data.json'
    dataset.write_text(json.dumps({'records': records}), encoding='utf-8')
    config = tmp_path / 'config.json'
    config.write_text('{}', encoding='utf-8')
    output = tmp_path / 'result.json'
    cfg = ModelConfig('openai_compatible', 'https://example.test', 'secret', 'm')
    monkeypatch.setattr(module, 'load_model_config', lambda *a: cfg)
    monkeypatch.setattr(module, 'OpenAICompatibleClient', lambda cfg: object())
    monkeypatch.setattr(module, 'source_hashes', lambda: {'source.py': 'fixed'})
    calls = []
    def fake_run(store, emb, gen, record, *, strategy, **kwargs):
        calls.append((strategy, record['id']))
        if len(calls) == 2: raise KeyboardInterrupt()
        return {'record_id': record['id'], 'conversation_id': record['conversation_id'], 'strategy': strategy,
                'metrics': {'answer': None, 'retrieval_hit': 1, 'reciprocal_rank': 1, 'context_hit': 1},
                'context_chars_used': 1, 'evidence': [], 'has_valid_citation': False,
                'invalid_citations': [], 'timings_seconds': {'total': 0}}
    monkeypatch.setattr(module, 'run_record', fake_run)
    argv = ['runner', str(dataset), str(output), '--index', idx, '--db', str(tmp_path / 'db.sqlite'), '--config', str(config), '--backend', 'exact']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(KeyboardInterrupt): module.main()
    assert len((tmp_path / 'result.json.jsonl').read_text(encoding='utf-8').splitlines()) == 2
    monkeypatch.setattr(sys, 'argv', argv + ['--resume'])
    module.main()
    result = json.loads(output.read_text(encoding='utf-8'))
    assert len(result['results']) == 6
    assert calls.count(('current', '0')) == 1
    assert all(s['succeeded'] == 2 for s in result['summary'].values())


def test_bootstrap_pairs_by_conversation_and_ignores_missing():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_rewrite_study.py'
    spec = importlib.util.spec_from_file_location('rewrite_analysis_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {'record_id': '1', 'conversation_id': 'a', 'strategy': 'current', 'metrics': {'retrieval_hit': 0}},
        {'record_id': '1', 'conversation_id': 'a', 'strategy': 'rewrite', 'metrics': {'retrieval_hit': 1}},
        {'record_id': '2', 'conversation_id': 'b', 'strategy': 'current', 'metrics': {'retrieval_hit': 1}},
        {'record_id': '2', 'conversation_id': 'b', 'strategy': 'rewrite', 'error': 'ProviderError'},
    ]
    x = module.paired_bootstrap(rows, 'rewrite', 'current', 'hit_at_k', 42, 100)
    assert x['n_pairs'] == 1 and x['n_conversations'] == 1
    assert x['difference'] == 1 and x['ci_95'] == [1, 1]
