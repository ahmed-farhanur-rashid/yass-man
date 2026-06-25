"""Unit tests for ResultCache."""

import time

import pytest

from backend.cache.result_cache import ResultCache, _normalise_key
from backend.schemas.result import LatencyBreakdown, RankedResult, SearchResponse


def _make_response(query: str = "test") -> SearchResponse:
    return SearchResponse(
        query_id="uuid-1",
        query=query,
        expanded_queries=[query],
        results=[],
        latency_ms=LatencyBreakdown(),
    )


# ── key normalisation ─────────────────────────────────────────────────────────

def test_normalise_lowercases():
    assert _normalise_key("GPU AI") == "gpu ai"

def test_normalise_collapses_whitespace():
    assert _normalise_key("gpu  ai") == "gpu ai"
    assert _normalise_key("  gpu ai  ") == "gpu ai"

def test_normalise_mixed():
    assert _normalise_key("  GPU   AI  ") == _normalise_key("gpu ai")


# ── basic get/set ─────────────────────────────────────────────────────────────

def test_miss_returns_none():
    cache = ResultCache()
    assert cache.get("anything") is None

def test_set_then_get():
    cache = ResultCache()
    resp = _make_response("test")
    cache.set("test", resp)
    assert cache.get("test") is resp

def test_case_insensitive_hit():
    cache = ResultCache()
    resp = _make_response("gpu ai")
    cache.set("GPU AI", resp)
    assert cache.get("gpu ai") is resp
    assert cache.get("GPU  AI") is resp

def test_size_increments():
    cache = ResultCache()
    assert cache.size == 0
    cache.set("q1", _make_response())
    assert cache.size == 1
    cache.set("q2", _make_response())
    assert cache.size == 2


# ── TTL ───────────────────────────────────────────────────────────────────────

def test_expired_entry_returns_none(monkeypatch):
    cache = ResultCache(ttl_seconds=1.0)
    cache.set("query", _make_response())

    # Fast-forward monotonic clock by 2 seconds
    original = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original() + 2.0)

    assert cache.get("query") is None

def test_unexpired_entry_returned(monkeypatch):
    cache = ResultCache(ttl_seconds=60.0)
    resp = _make_response()
    cache.set("query", resp)
    # No time skip → should still be valid
    assert cache.get("query") is resp

def test_zero_ttl_disables_expiry():
    cache = ResultCache(ttl_seconds=0.0)
    resp = _make_response()
    cache.set("query", resp)
    # ttl=0 means no expiry check
    assert cache.get("query") is resp


# ── LRU eviction ─────────────────────────────────────────────────────────────

def test_evicts_oldest_when_full():
    cache = ResultCache(max_size=3, ttl_seconds=0)
    for i in range(3):
        cache.set(f"q{i}", _make_response(f"q{i}"))
    # q0 is oldest; adding q3 should evict q0
    cache.set("q3", _make_response("q3"))
    assert cache.size == 3
    assert cache.get("q0") is None
    assert cache.get("q3") is not None

def test_access_refreshes_lru_order():
    cache = ResultCache(max_size=3, ttl_seconds=0)
    for i in range(3):
        cache.set(f"q{i}", _make_response(f"q{i}"))
    # Access q0 to make it recently used
    cache.get("q0")
    # Now q1 is the oldest → q1 should be evicted when q3 added
    cache.set("q3", _make_response("q3"))
    assert cache.get("q0") is not None
    assert cache.get("q1") is None


# ── invalidate / clear ────────────────────────────────────────────────────────

def test_invalidate_removes_entry():
    cache = ResultCache()
    cache.set("query", _make_response())
    cache.invalidate("query")
    assert cache.get("query") is None
    assert cache.size == 0

def test_invalidate_nonexistent_is_noop():
    cache = ResultCache()
    cache.invalidate("does not exist")  # should not raise

def test_clear_empties_cache():
    cache = ResultCache()
    for i in range(5):
        cache.set(f"q{i}", _make_response())
    cache.clear()
    assert cache.size == 0
    assert cache.get("q0") is None

def test_stats_reports_correctly():
    cache = ResultCache(max_size=10, ttl_seconds=30.0)
    stats = cache.stats()
    assert stats["max_size"] == 10
    assert stats["ttl_seconds"] == 30.0
    assert stats["size"] == 0
