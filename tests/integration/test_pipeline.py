"""
Integration tests for the full pipeline (mocked HTTP + models).

These tests wire up all pipeline stages with lightweight mocks so they
can run without ML models or a live SearXNG instance.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.cache.embedding_cache import EmbeddingCache
from backend.model_config_loader import (
    EmbedderConfig,
    PipelineConfig,
    RerankerConfig,
    RouterConfig,
)
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.expander import QueryExpander
from backend.pipeline.router import QueryRouter


def test_full_router_expander_pipeline():
    """Router + expander should produce a sensible list for a research query."""
    router = QueryRouter(RouterConfig(mode="rule-based"))
    expander = QueryExpander(PipelineConfig(max_expanded_queries=5))

    query = "best Python web frameworks 2026"
    rr = router.route(query)
    assert rr.intent == "research"

    expanded = expander.expand(query, rr)
    assert expanded[0] == query
    assert len(expanded) >= 2
    assert len(expanded) <= 6


def test_router_expander_aggregator_chain():
    """Simulate raw retrieval results going through aggregator."""
    from backend.schemas.result import RawResult

    agg = Aggregator()
    raw = [
        RawResult(title="FastAPI Docs", url="https://fastapi.tiangolo.com/", snippet="Modern web framework", query_source="fastapi tutorial"),
        RawResult(title="FastAPI Docs", url="https://fastapi.tiangolo.com/?utm_source=google", snippet="Modern web framework.", query_source="fastapi docs"),
        RawResult(title="Django Home", url="https://www.djangoproject.com/", snippet="The web framework for perfectionists", query_source="django tutorial"),
        RawResult(title="Flask Docs", url="https://flask.palletsprojects.com/", snippet="Micro framework", query_source="flask tutorial"),
    ]
    aggregated = agg.aggregate(raw)
    # fastapi.tiangolo.com appears twice (with UTM) → should merge to 1
    urls = [r.url for r in aggregated]
    domains = set(r.split("//")[1].split("/")[0].replace("www.", "") for r in urls)
    assert "fastapi.tiangolo.com" in domains
    assert len(aggregated) == 3  # fastapi (merged), django, flask


def test_embedding_cache_miss_then_hit():
    """EmbeddingCache should miss first time and hit second time."""
    import numpy as np
    cache = EmbeddingCache()
    url = "https://example.com/"
    assert cache.get(url) is None
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.set(url, emb)
    result = cache.get(url)
    assert result is not None
    assert (result == emb).all()
    assert cache.hit_rate == 0.5  # 1 miss, 1 hit
