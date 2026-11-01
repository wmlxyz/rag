"""Provider-independent model contracts."""
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


@dataclass(frozen=True)
class RankedItem:
    index: int
    score: float


class Generator(Protocol):
    def generate(self, messages: Sequence[Message], *, temperature: float = 0.0,
                 max_tokens: int | None = None) -> GenerationResult: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> list[RankedItem]: ...
