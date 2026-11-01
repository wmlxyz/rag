import io
import json
from urllib.error import HTTPError, URLError
import pytest
from rag.config import ModelConfig
from rag.errors import ProviderError, ProviderResponseError
from rag.models import Message
from rag.providers.openai_compatible import OpenAICompatibleClient


def client(retries=0):
    return OpenAICompatibleClient(ModelConfig("openai_compatible", "https://example.test/v1", "secret", "model-a", max_retries=retries))


def response(value):
    return io.BytesIO(json.dumps(value).encode())


def test_chat_request_and_response(monkeypatch):
    def fake_open(request, timeout):
        assert request.full_url == "https://example.test/v1/chat/completions"
        assert json.loads(request.data)["messages"][0]["content"] == "hello"
        assert request.get_header("Authorization") == "Bearer secret"
        return response({"model": "model-a", "choices": [{"message": {"content": "world"}, "finish_reason": "stop"}], "usage": {"total_tokens": 7}})
    monkeypatch.setattr("rag.providers.openai_compatible._open", fake_open)
    result = client().generate([Message("user", "hello")])
    assert (result.text, result.usage.total_tokens, result.finish_reason) == ("world", 7, "stop")


def test_embeddings_sorted_and_checked(monkeypatch):
    monkeypatch.setattr("rag.providers.openai_compatible._open", lambda *a, **k: response({"data": [{"index": 1, "embedding": [2, 3]}, {"index": 0, "embedding": [0, 1]}]}))
    assert client().embed(["a", "b"]) == [[0, 1], [2, 3]]
    monkeypatch.setattr("rag.providers.openai_compatible._open", lambda *a, **k: response({"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]}))
    with pytest.raises(ProviderResponseError):
        client().embed(["a", "b"])


def test_retry_and_redacted_error(monkeypatch):
    attempts = []
    def fail(request, timeout):
        attempts.append(1)
        raise HTTPError(request.full_url, 429, "secret response", {}, None)
    monkeypatch.setattr("rag.providers.openai_compatible._open", fail)
    monkeypatch.setattr("rag.providers.openai_compatible.time.sleep", lambda _: None)
    with pytest.raises(ProviderError) as exc:
        client(2).generate([Message("user", "hello")])
    assert len(attempts) == 3
    assert "secret" not in str(exc.value)


def test_no_retry_for_authentication_and_connection_error(monkeypatch):
    attempts = []
    def unauthorized(request, timeout):
        attempts.append(1)
        raise HTTPError(request.full_url, 401, "private", {}, None)
    monkeypatch.setattr("rag.providers.openai_compatible._open", unauthorized)
    with pytest.raises(ProviderError, match="401"):
        client(2).generate([Message("user", "x")])
    assert len(attempts) == 1
    monkeypatch.setattr("rag.providers.openai_compatible._open", lambda *a, **k: (_ for _ in ()).throw(URLError("private")))
    with pytest.raises(ProviderError) as exc:
        client().generate([Message("user", "x")])
    assert "private" not in str(exc.value)


def test_malformed_chat(monkeypatch):
    monkeypatch.setattr("rag.providers.openai_compatible._open", lambda *a, **k: response({"choices": []}))
    with pytest.raises(ProviderResponseError):
        client().generate([Message("user", "x")])
