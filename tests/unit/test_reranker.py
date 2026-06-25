"""Unit tests for the reranker pipeline stage (model is mocked)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.model_config_loader import RerankerConfig
from backend.pipeline.reranker import RerankerStage
from backend.schemas.result import AggregatedResult


def _make_aggregated(url: str, title: str, snippet: str) -> AggregatedResult:
    return AggregatedResult(
        title=title,
        snippet=snippet,
        url=url,
        normalized_url=url,
    )


@pytest.fixture
def mock_reranker_model():
    model = MagicMock()
    # Return decreasing scores so first doc is best
    model.score.side_effect = lambda query, docs: [1.0 - i * 0.1 for i in range(len(docs))]
    model.model_name = "mock-reranker"
    return model


@pytest.fixture
def config():
    return RerankerConfig(
        model="mock",
        source="huggingface",
        repo="mock/repo",
        backend="pytorch",
        top_k=5,
    )


@pytest.fixture
def stage(mock_reranker_model, config):
    return RerankerStage(mock_reranker_model, config)


def test_rerank_sorts_descending(stage):
    results = [
        _make_aggregated("https://a.com", "A", "doc A"),
        _make_aggregated("https://b.com", "B", "doc B"),
        _make_aggregated("https://c.com", "C", "doc C"),
    ]
    ranked = asyncio.run(
        stage.rerank("test query", results)
    )
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_top_k(stage):
    results = [
        _make_aggregated(f"https://site{i}.com", f"Title {i}", f"Snippet {i}")
        for i in range(10)
    ]
    ranked = asyncio.run(
        stage.rerank("test query", results)
    )
    assert len(ranked) <= 5


def test_rerank_empty_input(stage):
    ranked = asyncio.run(
        stage.rerank("test query", [])
    )
    assert ranked == []


def test_rerank_result_has_score(stage):
    results = [_make_aggregated("https://x.com", "X", "snippet")]
    ranked = asyncio.run(
        stage.rerank("query", results)
    )
    assert isinstance(ranked[0].score, float)


def test_rerank_preserves_metadata(stage):
    results = [
        _make_aggregated("https://example.com", "Example Title", "Example snippet text here")
    ]
    ranked = asyncio.run(
        stage.rerank("query", results)
    )
    assert ranked[0].title == "Example Title"
    assert ranked[0].url == "https://example.com"


def test_rerank_passes_through_query_sources(stage):
    results = [
        AggregatedResult(
            title="A", snippet="snippet a", url="https://a.com/",
            normalized_url="https://a.com/",
            query_sources=["original query", "expanded query"],
        )
    ]
    ranked = asyncio.run(stage.rerank("test query", results))
    assert ranked[0].query_sources == ["original query", "expanded query"]
