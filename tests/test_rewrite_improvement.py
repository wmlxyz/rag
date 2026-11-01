from rag.errors import ProviderError
from rag.models import GenerationResult
from rag.pipeline import Hit, Store, answer, reciprocal_rank_fusion, validate_rewrite
from rag.rewrite_checks import check_rewrite
import json
from pathlib import Path


def test_rrf_deduplicates_and_is_deterministic():
    a = Hit("a", "a", 0, 1, "a", .8)
    b = Hit("b", "b", 0, 1, "b", .7)
    c = Hit("c", "c", 0, 1, "c", .9)
    fused = reciprocal_rank_fusion([[a, b], [c, b]], 3)
    assert [x.chunk_id for x in fused] == ["b", "a", "c"]
    assert fused[0].score > fused[1].score
    assert not validate_rewrite("line one\nline two", "question")
    assert not validate_rewrite("", "question")


def test_fusion_and_malformed_fallback(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "original.txt").write_text("apple", encoding="utf-8")
    (docs / "rewrite.txt").write_text("banana", encoding="utf-8")
    class Embedding:
        def embed(self, texts):
            return [[1., 0.] if "apple" in x else [0., 1.] for x in texts]
    class Generator:
        def __init__(self, rewrite): self.rewrite = rewrite
        def generate(self, messages, **kwargs):
            if "Rewrite" in messages[0].content:
                return GenerationResult(self.rewrite, "fake")
            return GenerationResult("answer [1]", "fake")
    store = Store(tmp_path / "db")
    idx = store.build(docs, Embedding(), embedding_id="v1", size=100, overlap=0)["index_id"]
    store.save_turn("s", "earlier", "answer")
    result = answer(store, Embedding(), Generator("banana"), index_id=idx, embedding_id="v1", question="apple", session_id="s", strategy="rewrite_fusion", top_k=1, persist=False)
    assert result["score_type"] == "rrf"
    assert set(result["retrieval_branches"]) == {"original", "rewritten"}
    assert result["retrieval_branches"]["original"] != result["retrieval_branches"]["rewritten"]
    bad = answer(store, Embedding(), Generator("bad\noutput"), index_id=idx, embedding_id="v1", question="apple", session_id="s", strategy="rewrite_fusion", top_k=1, persist=False)
    assert bad["query"] == "apple" and bad["rewrite_error"]
    assert bad["score_type"] == "cosine"


def test_checks_obvious_errors_and_do_not_claim_semantic_proof():
    assert 'question_to_nonquestion' in check_rewrite('What is one of her albums?', 'Concentration 20')['reasons']
    assert 'number_omitted' in check_rewrite('What happened in 1999?', 'What happened to Alice?')['reasons']
    assert 'negation_omitted' in check_rewrite('Did he not attend?', 'Did Alex attend?')['reasons']
    good = check_rewrite('What did she do?', 'What did Alice do?')
    assert good['passed'] and good['semantic_fidelity'] == 'not_verified'
    assert check_rewrite('她是什么科？', '苹果属于什么科？')['passed']


def test_checked_rewrite_falls_back_to_history_for_answer_like_output(tmp_path):
    docs = tmp_path / 'docs'; docs.mkdir()
    (docs / 'doc.txt').write_text('Concentration 20', encoding='utf-8')
    class Embedding:
        def embed(self, texts): return [[1., 0.] for _ in texts]
    class Generator:
        def generate(self, messages, **kwargs):
            return GenerationResult('Concentration 20' if 'Rewrite' in messages[0].content else 'Answer [1]', 'fake')
    store = Store(tmp_path / 'db')
    idx = store.build(docs, Embedding(), embedding_id='v1', size=100, overlap=0)['index_id']
    store.save_turn('s', 'Tell me about the musician', 'She released music.')
    result = answer(store, Embedding(), Generator(), index_id=idx, embedding_id='v1', session_id='s',
                    question='What is one of her albums?', strategy='rewrite_checked_fusion', persist=False)
    assert result['rewrite_fallback'] == 'history'
    assert result['rewrite_candidate'] == 'Concentration 20'
    assert 'question_to_nonquestion' in result['rewrite_check']['reasons']
    assert 'Tell me about the musician' in result['query']
    assert result['score_type'] == 'cosine' and set(result['retrieval_branches']) == {'query'}


def test_review_cases_show_scope_and_known_miss():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'evaluation' / 'rewrite_review.json'
    for case in json.loads(path.read_text(encoding='utf-8'))['cases']:
        check = check_rewrite(case['question'], case['rewrite'])
        if case['rule_expected'] == 'pass':
            assert check['passed']
        else:
            assert case['rule_expected'] in check['reasons']


def test_model_verifier_pass_reject_uncertain_and_invalid(tmp_path):
    docs = tmp_path / 'docs'; docs.mkdir()
    (docs / 'doc.txt').write_text('Alice released an album.', encoding='utf-8')
    class Embedding:
        def embed(self, texts): return [[1.] for _ in texts]
    class Generator:
        def __init__(self, decision): self.decision = decision
        def generate(self, messages, **kwargs):
            system = messages[0].content
            if 'Rewrite' in system: return GenerationResult('What did Alice release?', 'fake')
            if 'Assess whether' in system:
                if isinstance(self.decision, Exception): raise self.decision
                return GenerationResult(self.decision, 'fake')
            return GenerationResult('An album [1]', 'fake')
    store = Store(tmp_path / 'db')
    idx = store.build(docs, Embedding(), embedding_id='v1', size=100, overlap=0)['index_id']
    store.save_turn('s', 'Tell me about Alice', 'She is a singer.')
    for output, expected in [('{"decision":"pass"}', 'pass'), ('{"decision":"reject"}', 'reject'),
                              ('{"decision":"uncertain"}', 'uncertain'), ('invalid', 'error'), (ProviderError('offline'), 'error')]:
        result = answer(store, Embedding(), Generator(output), index_id=idx, embedding_id='v1', session_id='s',
                        question='What did she release?', strategy='rewrite_verified', persist=False)
        assert result['rewrite_verdict'] == expected
        assert result['rewrite_fallback'] == (None if expected == 'pass' else 'history')
        assert result['query'] == ('What did Alice release?' if expected == 'pass' else 'user: Tell me about Alice\nassistant: She is a singer.\nuser: What did she release?')
