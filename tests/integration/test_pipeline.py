"""
Integration tests: multiple pipeline stages wired together with mocks.

These tests exercise stage-to-stage data flow without ML models or a live
SearXNG instance. They catch schema mismatches and ordering bugs that unit
tests miss.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from backend.cache.embedding_cache import EmbeddingCache
from backend.model_config_loader import PipelineConfig, RerankerConfig, RouterConfig
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.embedder import EmbedderStage
from backend.pipeline.expander import QueryExpander
from backend.pipeline.reranker import RerankerStage
from backend.pipeline.router import QueryRouter
from backend.schemas.result import AggregatedResult, RawResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return QueryRouter(RouterConfig(mode="rule-based"))

@pytest.fixture
def expander():
    return QueryExpander(PipelineConfig(max_expanded_queries=5))

@pytest.fixture
def aggregator():
    return Aggregator()

@pytest.fixture
def embedder_stage():
    """EmbedderStage backed by a mock model that returns orthogonal unit vectors."""
    model = MagicMock()
    def _embed(texts):
        n = len(texts)
        arr = np.eye(n, dtype=np.float32)  # orthogonal → never cluster
        return arr
    model.embed.side_effect = _embed
    return EmbedderStage(
        model=model,
        cache=EmbeddingCache(),
        config=PipelineConfig(dedup_similarity_threshold=0.92),
        enabled=True,
    )

@pytest.fixture
def reranker_stage():
    model = MagicMock()
    model.score.side_effect = lambda q, docs: [1.0 - i * 0.05 for i in range(len(docs))]
    return RerankerStage(model, RerankerConfig(
        model="mock", source="hf", repo="mock/repo", backend="pytorch", top_k=5
    ))


# ── Router → Expander ─────────────────────────────────────────────────────────

def test_research_query_expands_correctly(router, expander):
    rr = router.route("best Python web frameworks 2026")
    assert rr.intent == "research"
    expanded = expander.expand("best Python web frameworks 2026", rr)
    assert expanded[0] == "best Python web frameworks 2026"
    assert len(expanded) >= 2

def test_fact_query_expands_fewer(router, expander):
    rr = router.route("what is entropy")
    assert rr.intent == "fact"
    expanded = expander.expand("what is entropy", rr)
    assert len(expanded) <= 3

def test_conversational_stays_minimal(router, expander):
    rr = router.route("hello")
    expanded = expander.expand("hello", rr)
    assert len(expanded) <= 2


# ── Aggregator dedup ──────────────────────────────────────────────────────────

def test_aggregator_merges_utm_duplicates(aggregator):
    raw = [
        RawResult(title="FastAPI", url="https://fastapi.tiangolo.com/", snippet="Web framework", query_source="q1"),
        RawResult(title="FastAPI", url="https://fastapi.tiangolo.com/?utm_source=google", snippet="Fast.", query_source="q2"),
        RawResult(title="Django", url="https://www.djangoproject.com/", snippet="Perfectionists", query_source="q1"),
    ]
    out = aggregator.aggregate(raw)
    assert len(out) == 2
    fastapi = next(r for r in out if "fastapi" in r.url)
    assert "q1" in fastapi.query_sources
    assert "q2" in fastapi.query_sources

def test_aggregator_preserves_order_of_first_seen(aggregator):
    raw = [
        RawResult(title="First", url="https://first.com/", snippet="s", query_source="q"),
        RawResult(title="Second", url="https://second.com/", snippet="s", query_source="q"),
        RawResult(title="Third", url="https://third.com/", snippet="s", query_source="q"),
    ]
    out = aggregator.aggregate(raw)
    titles = [r.title for r in out]
    assert titles == ["First", "Second", "Third"]


# ── Aggregator → Embedder ─────────────────────────────────────────────────────

def test_embedder_stage_passes_through_with_orthogonal_embeddings(aggregator, embedder_stage):
    raw = [
        RawResult(title=f"Site {i}", url=f"https://site{i}.com/", snippet=f"content {i}", query_source="q")
        for i in range(4)
    ]
    agg = aggregator.aggregate(raw)
    clustered = asyncio.run(embedder_stage.cluster(agg))
    # All orthogonal → no merging
    assert len(clustered) == len(agg)

def test_embedder_reduces_near_duplicates():
    """Stage with threshold=0.0 clusters everything together."""
    model = MagicMock()
    # All results get the same embedding → sim=1.0 everywhere
    model.embed.return_value = np.ones((3, 4), dtype=np.float32)
    stage = EmbedderStage(
        model=model,
        cache=EmbeddingCache(),
        config=PipelineConfig(dedup_similarity_threshold=0.5),
        enabled=True,
    )
    results = [
        AggregatedResult(title=f"T{i}", snippet="x" * (i + 1), url=f"https://s{i}.com/", normalized_url=f"https://s{i}.com/")
        for i in range(3)
    ]
    out = asyncio.run(stage.cluster(results))
    assert len(out) == 1
    # Longest snippet wins
    assert out[0].snippet == "xxx"


# ── Embedder → Reranker ───────────────────────────────────────────────────────

def test_reranker_sorts_descending(aggregator, embedder_stage, reranker_stage):
    raw = [
        RawResult(title=f"Result {i}", url=f"https://r{i}.com/", snippet=f"content about topic {i}", query_source="q")
        for i in range(6)
    ]
    agg = aggregator.aggregate(raw)
    clustered = asyncio.run(embedder_stage.cluster(agg))
    ranked = asyncio.run(reranker_stage.rerank("topic query", clustered))
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)

def test_reranker_respects_top_k(aggregator, embedder_stage, reranker_stage):
    raw = [
        RawResult(title=f"R{i}", url=f"https://r{i}.com/", snippet=f"s{i}", query_source="q")
        for i in range(10)
    ]
    agg = aggregator.aggregate(raw)
    clustered = asyncio.run(embedder_stage.cluster(agg))
    ranked = asyncio.run(reranker_stage.rerank("query", clustered))
    assert len(ranked) <= 5  # top_k=5 in fixture

def test_ranked_results_carry_url(aggregator, embedder_stage, reranker_stage):
    raw = [RawResult(title="T", url="https://target.com/", snippet="relevant content", query_source="q")]
    agg = aggregator.aggregate(raw)
    clustered = asyncio.run(embedder_stage.cluster(agg))
    ranked = asyncio.run(reranker_stage.rerank("query", clustered))
    assert ranked[0].url == "https://target.com/"
