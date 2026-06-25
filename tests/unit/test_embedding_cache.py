"""Unit tests for EmbeddingCache."""

import numpy as np
import pytest

from backend.cache.embedding_cache import EmbeddingCache


@pytest.fixture
def cache() -> EmbeddingCache:
    return EmbeddingCache()


def test_miss_returns_none(cache):
    assert cache.get("https://never-seen.com/") is None


def test_set_then_get(cache):
    url = "https://example.com/page"
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.set(url, emb)
    result = cache.get(url)
    assert result is not None
    assert np.allclose(result, emb)


def test_different_urls_stored_separately(cache):
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    cache.set("https://a.com/", e1)
    cache.set("https://b.com/", e2)
    assert np.allclose(cache.get("https://a.com/"), e1)
    assert np.allclose(cache.get("https://b.com/"), e2)


def test_overwrite_updates_value(cache):
    url = "https://example.com/"
    cache.set(url, np.array([1.0], dtype=np.float32))
    cache.set(url, np.array([2.0], dtype=np.float32))
    assert cache.get(url)[0] == pytest.approx(2.0)


def test_size_tracks_entries(cache):
    assert cache.size == 0
    cache.set("https://a.com/", np.zeros(3, dtype=np.float32))
    assert cache.size == 1
    cache.set("https://b.com/", np.zeros(3, dtype=np.float32))
    assert cache.size == 2


def test_hit_rate_zero_on_empty(cache):
    assert cache.hit_rate == 0.0


def test_hit_rate_after_miss_and_hit(cache):
    url = "https://x.com/"
    cache.get(url)          # miss
    cache.set(url, np.zeros(2, dtype=np.float32))
    cache.get(url)          # hit
    assert cache.hit_rate == pytest.approx(0.5)


def test_stats_structure(cache):
    stats = cache.stats()
    assert set(stats.keys()) == {"size", "hits", "misses", "hit_rate"}


def test_clear_resets_everything(cache):
    cache.set("https://a.com/", np.zeros(2, dtype=np.float32))
    cache.get("https://a.com/")
    cache.clear()
    assert cache.size == 0
    assert cache.hit_rate == 0.0
    assert cache.get("https://a.com/") is None
