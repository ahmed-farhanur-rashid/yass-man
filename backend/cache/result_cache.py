"""
Result Cache.

In-memory LRU cache for complete search responses, keyed by normalised query
string. Avoids redundant SearXNG round-trips and model inference for repeated
queries within a session.

Design:
- Max 256 entries (configurable). Oldest entry evicted when full.
- Keys are lower-cased, whitespace-collapsed query strings.
- Not persisted across restarts — intentional, results age quickly.
- Thread-safe for asyncio (single-threaded event loop, no locks needed).
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Optional

from backend.schemas.result import SearchResponse


def _normalise_key(query: str) -> str:
    """Lower-case and collapse whitespace so 'GPU AI' == 'gpu  ai'."""
    return re.sub(r"\s+", " ", query.strip().lower())


class ResultCache:
    """LRU cache for SearchResponse objects."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        # OrderedDict preserves insertion order for LRU eviction.
        # Each value is (response, inserted_at).
        self._store: OrderedDict[str, tuple[SearchResponse, float]] = OrderedDict()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[SearchResponse]:
        """Return a cached response or None if missing / expired."""
        key = _normalise_key(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        response, inserted_at = entry
        if self._ttl > 0 and (time.monotonic() - inserted_at) > self._ttl:
            del self._store[key]
            return None
        # Move to end (most-recently-used).
        self._store.move_to_end(key)
        return response

    def set(self, query: str, response: SearchResponse) -> None:
        """Store a response. Evicts the oldest entry when at capacity."""
        key = _normalise_key(query)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (response, time.monotonic())
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict LRU

    def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        key = _normalise_key(query)
        self._store.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache."""
        self._store.clear()

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }
