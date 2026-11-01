"""DashScope native text reranking adapter."""
import math
from ..errors import ProviderResponseError
from ..models import RankedItem
from .openai_compatible import OpenAICompatibleClient


class DashScopeReranker:
    def __init__(self, config):
        if config.provider != "dashscope_rerank":
            raise ValueError("Expected dashscope_rerank provider")
        self.config = config
        self.client = OpenAICompatibleClient(config)
        self.last_usage = None

    def rerank(self, query, documents):
        self.last_usage = None
        if not query.strip() or not documents or len(documents) > 500:
            raise ValueError("Reranking requires a query and 1–500 documents")
        data = self.client._post("/services/rerank/text-rerank/text-rerank", {
            "model": self.config.model, "input": {"query": query, "documents": list(documents)},
            "parameters": {"top_n": len(documents)}})
        try:
            items = data["output"]["results"]
            if not isinstance(items, list) or len(items) != len(documents):
                raise ValueError("incomplete rerank result")
            result = []
            for item in items:
                index, score = item["index"], item["relevance_score"]
                if type(index) is not int or not 0 <= index < len(documents) or type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("invalid rerank result")
                result.append(RankedItem(index, float(score)))
            if len({item.index for item in result}) != len(documents):
                raise ValueError("duplicate rerank index")
            self.last_usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            return sorted(result, key=lambda item: (-item.score, item.index))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError("Provider returned invalid reranking results") from exc
