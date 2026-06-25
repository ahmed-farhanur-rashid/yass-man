"""
Shared pytest fixtures.

Lightweight only — no ML model loading. Tests that need mocked models
build them inline to keep dependencies explicit.
"""

from __future__ import annotations

import pytest

from backend.cache.embedding_cache import EmbeddingCache
from backend.cache.result_cache import ResultCache
from backend.model_config_loader import PipelineConfig, RerankerConfig, RouterConfig
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.expander import QueryExpander
from backend.pipeline.router import QueryRouter


@pytest.fixture(scope="session")
def router_config() -> RouterConfig:
    return RouterConfig(mode="rule-based")


@pytest.fixture(scope="session")
def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        max_expanded_queries=5,
        search_timeout_seconds=2.0,
        dedup_similarity_threshold=0.92,
        embedding_cache=True,
    )


@pytest.fixture(scope="session")
def router(router_config) -> QueryRouter:
    return QueryRouter(router_config)


@pytest.fixture(scope="session")
def expander(pipeline_config) -> QueryExpander:
    return QueryExpander(pipeline_config)


@pytest.fixture(scope="session")
def aggregator() -> Aggregator:
    return Aggregator()


@pytest.fixture
def embedding_cache() -> EmbeddingCache:
    """Fresh cache per test."""
    return EmbeddingCache()


@pytest.fixture
def result_cache() -> ResultCache:
    """Fresh result cache per test."""
    return ResultCache()
