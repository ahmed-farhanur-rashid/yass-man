"""
Embedding Cache — Phase 5.

In-memory cache for computed embeddings, keyed by normalized URL.
Embeddings are deterministic so no TTL is needed.
Thread-safe for asyncio (single-threaded event loop).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple dict-based in-memory cache for URL embeddings."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0

    def get(self, url: str) -> Optional[np.ndarray]:
        embedding = self._cache.get(url)
        if embedding is not None:
            self._hits += 1
            return embedding
        self._misses += 1
        return None

    def set(self, url: str, embedding: np.ndarray) -> None:
        self._cache[url] = embedding

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
        }
