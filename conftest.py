"""
Shared pytest fixtures available across all test modules.

Model-heavy fixtures (embedder, reranker, llm) are NOT loaded here by default
to keep the unit test suite fast. Integration tests that need them should
import or mock them directly.
"""

from __future__ import annotations

import pytest

from backend.cache.embedding_cache import EmbeddingCache
from backend.model_config_loader import (
    EmbedderConfig,
    LLMConfig,
    PipelineConfig,
    RerankerConfig,
    RouterConfig,
)
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.expander import QueryExpander
from backend.pipeline.router import QueryRouter


# ── Config fixtures ───────────────────────────────────────────────────────────


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
def reranker_config() -> RerankerConfig:
    return RerankerConfig(
        model="bge-reranker-base",
        source="huggingface",
        repo="BAAI/bge-reranker-base",
        backend="pytorch",
        top_k=10,
    )


# ── Pipeline component fixtures ───────────────────────────────────────────────


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
    """Fresh cache per test (function scope)."""
    return EmbeddingCache()
