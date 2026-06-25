"""Unit tests for EmbedderStage clustering logic (embedding model is mocked)."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.cache.embedding_cache import EmbeddingCache
from backend.model_config_loader import PipelineConfig
from backend.pipeline.embedder import EmbedderStage
from backend.schemas.result import AggregatedResult


def _make_result(url: str, snippet: str = "snippet") -> AggregatedResult:
    return AggregatedResult(
        title=f"Title for {url}",
        snippet=snippet,
        url=url,
        normalized_url=url,
    )


def _make_stage(embeddings: list[list[float]], threshold: float = 0.92) -> EmbedderStage:
    """
    Build a stage whose model returns the given embedding rows in order.
    Embeddings are L2-normalised so cosine sim == dot product.
    """
    arr = np.array(embeddings, dtype=np.float32)
    # Normalise each row
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1, norms)

    model = MagicMock()
    model.embed.return_value = arr
    config = PipelineConfig(dedup_similarity_threshold=threshold)
    return EmbedderStage(model=model, cache=EmbeddingCache(), config=config, enabled=True)


# ── passthrough cases ─────────────────────────────────────────────────────────

def test_disabled_returns_unchanged():
    model = MagicMock()
    config = PipelineConfig()
    stage = EmbedderStage(model=model, cache=EmbeddingCache(), config=config, enabled=False)
    results = [_make_result(f"https://s{i}.com/") for i in range(5)]
    out = asyncio.run(stage.cluster(results))
    assert out == results
    model.embed.assert_not_called()


def test_single_result_returns_unchanged():
    stage = _make_stage([[1.0, 0.0]])
    results = [_make_result("https://only.com/")]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 1


def test_empty_list_returns_empty():
    model = MagicMock()
    config = PipelineConfig()
    stage = EmbedderStage(model=model, cache=EmbeddingCache(), config=config, enabled=True)
    out = asyncio.run(stage.cluster([]))
    assert out == []


# ── clustering logic ──────────────────────────────────────────────────────────

def test_identical_embeddings_merged_to_one():
    """Two results with identical embeddings (sim=1.0) → one representative."""
    stage = _make_stage([[1.0, 0.0], [1.0, 0.0]], threshold=0.92)
    results = [
        _make_result("https://a.com/", snippet="short"),
        _make_result("https://b.com/", snippet="much longer snippet wins"),
    ]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 1
    # Should keep the one with the longer snippet
    assert out[0].snippet == "much longer snippet wins"


def test_orthogonal_embeddings_not_merged():
    """Orthogonal embeddings (sim=0.0) stay separate."""
    stage = _make_stage([[1.0, 0.0], [0.0, 1.0]], threshold=0.92)
    results = [_make_result("https://a.com/"), _make_result("https://b.com/")]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 2


def test_three_clusters_reduce_correctly():
    """
    a ≈ b (high sim), c is separate.
    Expect 2 representatives.
    """
    # a and b are nearly identical; c is orthogonal
    embs = [
        [1.0, 0.01, 0.0],   # a
        [1.0, 0.01, 0.0],   # b (same as a)
        [0.0, 0.0, 1.0],    # c
    ]
    stage = _make_stage(embs, threshold=0.92)
    results = [
        _make_result("https://a.com/", snippet="aaa"),
        _make_result("https://b.com/", snippet="bbb longer"),
        _make_result("https://c.com/", snippet="ccc"),
    ]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 2
    urls = {r.url for r in out}
    assert "https://c.com/" in urls


def test_representative_has_longest_snippet():
    embs = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    stage = _make_stage(embs, threshold=0.5)
    results = [
        _make_result("https://a.com/", snippet="x"),
        _make_result("https://b.com/", snippet="much longer snippet here"),
        _make_result("https://c.com/", snippet="medium length snippet"),
    ]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 1
    assert out[0].snippet == "much longer snippet here"


# ── cache interaction ─────────────────────────────────────────────────────────

def test_cache_populated_after_cluster():
    model = MagicMock()
    arr = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model.embed.return_value = arr
    cache = EmbeddingCache()
    config = PipelineConfig(dedup_similarity_threshold=0.92)
    stage = EmbedderStage(model=model, cache=cache, config=config, enabled=True)

    results = [_make_result("https://a.com/"), _make_result("https://b.com/")]
    asyncio.run(stage.cluster(results))
    assert cache.get("https://a.com/") is not None
    assert cache.get("https://b.com/") is not None


def test_cached_embeddings_not_re_embedded():
    model = MagicMock()
    cache = EmbeddingCache()
    # Pre-populate cache with known embeddings
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    cache.set("https://a.com/", e1)
    cache.set("https://b.com/", e2)

    config = PipelineConfig(dedup_similarity_threshold=0.92)
    stage = EmbedderStage(model=model, cache=cache, config=config, enabled=True)
    results = [_make_result("https://a.com/"), _make_result("https://b.com/")]
    asyncio.run(stage.cluster(results))
    # Model should never be called — both were cached
    model.embed.assert_not_called()
