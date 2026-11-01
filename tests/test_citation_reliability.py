import importlib.util
import json
from pathlib import Path
import pytest
from rag.citations import inspect_citations
from rag.errors import ProviderError
from rag.models import GenerationResult, Usage
from rag.pipeline import Store, answer


class Embedding:
    def embed(self, texts): return [[1., 0.] if '苹果' in x else [0., 1.] for x in texts]


class Generator:
    def __init__(self, outputs): self.outputs, self.calls = iter(outputs), []
    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        result = next(self.outputs)
        if isinstance(result, Exception): raise result
        return GenerationResult(result, 'fake', Usage(total_tokens=10))


@pytest.fixture
def index(tmp_path):
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'a.txt').write_text('苹果属于蔷薇科。', encoding='utf-8')
    store = Store(tmp_path / 'db')
    idx = store.build(docs, Embedding(), embedding_id='v1', size=50, overlap=0)['index_id']
    return store, idx


def test_citation_parser_multiple_invalid_and_malformed():
    result = inspect_citations('内容[1, 2]，其它[9]，范围[1-3]。', {1, 2})
    assert result['valid_ids'] == [1, 2] and result['invalid_ids'] == [9]
    assert result['malformed'] == ['[1-3]'] and not result['passed']


def test_invalid_answer_is_repaired_and_logged(index):
    store, idx = index
    model = Generator(['苹果属于蔷薇科[99]。', '苹果属于蔷薇科[1]。'])
    result = answer(store, Embedding(), model, index_id=idx, embedding_id='v1', question='苹果属于什么科？')
    assert result['answer'] == '苹果属于蔷薇科[1]。'
    assert result['initial_citation_check']['invalid_ids'] == [99]
    assert result['citation_repair_status'] == 'repaired' and result['citation_repair_usage']['total_tokens'] == 10
    assert result['evidence_support_status'] == 'not_verified' and len(model.calls) == 2
    assert store.history(result['session_id'])[-1].content == result['answer']


@pytest.mark.parametrize('second', ['没有引用的回答', '仍然错误[42]', ProviderError('offline')])
def test_failed_repair_abstains_and_history_never_saves_invalid_answer(index, second):
    store, idx = index
    model = Generator(['错误回答[42]', second])
    result = answer(store, Embedding(), model, index_id=idx, embedding_id='v1', question='苹果是什么？')
    assert result['answer_status'] == 'citation_unresolved'
    assert '错误回答' not in result['answer'] and result['initial_citation_check']['invalid_ids'] == [42]
    assert store.history(result['session_id'])[-1].content == result['answer']


def test_warn_and_no_evidence_do_not_call_repair(index):
    store, idx = index
    model = Generator(['错误[9]'])
    result = answer(store, Embedding(), model, index_id=idx, embedding_id='v1', question='苹果', citation_policy='warn')
    assert result['invalid_citations'] == [9] and len(model.calls) == 1
    empty = Generator([])
    result = answer(store, Embedding(), empty, index_id=idx, embedding_id='v1', question='天气', min_score=.5)
    assert result['answer_status'] == 'no_evidence' and not empty.calls


def test_manual_review_examples_distinguish_valid_numbers_from_support():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'evaluation' / 'citation_review.json'
    cases = json.loads(path.read_text(encoding='utf-8'))['cases']
    for case in cases:
        check = inspect_citations(case['answer'], {item['id'] for item in case['evidence']})
        if case['citation_structure'] == 'valid':
            assert check['passed']
        elif case['citation_structure'] == 'invalid':
            assert check['invalid_ids']
        else:
            assert not check['has_valid']
    # An unsupported statement can still have a syntactically valid citation.
    assert any(c['citation_structure'] == 'valid' and c['human_support_label'] == 'unsupported' for c in cases)


def test_threshold_metrics_distinguish_false_accepts_and_rejects():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'calibrate_threshold.py'
    spec = importlib.util.spec_from_file_location('calibration_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'score': .9, 'answerable': True}, {'score': .4, 'answerable': True},
            {'score': .8, 'answerable': False}, {'score': .1, 'answerable': False}]
    result = module.metrics(rows, .5)
    assert (result['tp'], result['fn'], result['fp'], result['tn']) == (1, 1, 1, 1)
