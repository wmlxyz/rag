"""Bounded in-process cache for one configured embedding provider."""
from collections import OrderedDict
from threading import RLock


class CachedEmbedder:
    def __init__(self, provider, capacity=128):
        if capacity < 1:
            raise ValueError("Cache capacity must be positive")
        self.provider = provider
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = RLock()
        self.hits = self.misses = self.provider_calls = 0

    def embed(self, texts):
        if not texts:
            return []
        if any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        with self._lock:
            missing = []
            for text in texts:
                if text in self._entries:
                    self.hits += 1
                    self._entries.move_to_end(text)
                else:
                    self.misses += 1
                    if text not in missing:
                        missing.append(text)
            vectors = {}
            if missing:
                self.provider_calls += 1
                returned = self.provider.embed(missing)
                if len(returned) != len(missing):
                    raise ValueError("Embedding response count mismatch")
                vectors = {text: list(vector) for text, vector in zip(missing, returned)}
            result = [list(self._entries[t]) if t in self._entries else list(vectors[t]) for t in texts]
            for text in missing:
                self._entries[text] = tuple(vectors[text])
                self._entries.move_to_end(text)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
            return result

    def stats(self):
        with self._lock:
            return {"hits": self.hits, "misses": self.misses,
                    "provider_calls": self.provider_calls, "size": len(self._entries), "capacity": self.capacity}
