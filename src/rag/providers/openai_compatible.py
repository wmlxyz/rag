"""Minimal OpenAI-compatible chat and embeddings adapter using the standard library."""
import json
import logging
import math
import time
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..config import ModelConfig
from ..errors import ProviderError, ProviderResponseError
from ..models import GenerationResult, Message, Usage

logger = logging.getLogger(__name__)


class _NoRedirect(HTTPRedirectHandler):
    # Do not forward credentials to an unexpected redirect destination.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(request, timeout):
    return build_opener(_NoRedirect).open(request, timeout=timeout)


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def _post(self, endpoint: str, payload: dict) -> dict:
        request = Request(
            self.config.base_url.rstrip("/") + endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                with _open(request, timeout=self.config.timeout_seconds) as response:
                    data = json.load(response)
                if not isinstance(data, dict):
                    raise ProviderResponseError("Provider returned a non-object response")
                return data
            except HTTPError as exc:
                retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
                if not retryable or attempt == self.config.max_retries:
                    raise ProviderError(f"Provider HTTP error: {exc.code}") from None
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.config.max_retries:
                    raise ProviderError(f"Provider connection failed: {type(exc).__name__}") from None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProviderResponseError("Provider returned invalid JSON") from exc
            logger.warning("Provider request failed; retrying (%d/%d)", attempt + 1, self.config.max_retries)
            time.sleep(min(0.5 * 2 ** attempt, 4.0))
        raise ProviderError("Provider request failed")

    def generate(self, messages: Sequence[Message], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> GenerationResult:
        if not messages:
            raise ValueError("At least one message is required")
        payload = {"model": self.config.model, "messages": [{"role": m.role, "content": m.content} for m in messages], "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        data = self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("non-text completion")
            usage = data.get("usage") or {}
            return GenerationResult(text=content, model=data.get("model") or self.config.model,
                usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")),
                finish_reason=choice.get("finish_reason"))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError("Provider returned an invalid chat response") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        data = self._post("/embeddings", {"model": self.config.model, "input": list(texts)})
        try:
            items = data["data"]
            if len(items) != len(texts):
                raise ValueError("wrong item count")
            indexed = sorted(items, key=lambda item: item["index"])
            if [item["index"] for item in indexed] != list(range(len(texts))):
                raise ValueError("invalid indices")
            vectors = [item["embedding"] for item in indexed]
            dimension = len(vectors[0])
            if not dimension or any(not isinstance(v, list) or len(v) != dimension or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("invalid vectors")
            return vectors
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError("Provider returned invalid embeddings") from exc
