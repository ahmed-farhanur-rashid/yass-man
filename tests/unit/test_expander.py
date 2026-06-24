"""Unit tests for the query expander."""

import pytest
from backend.model_config_loader import PipelineConfig, RouterConfig
from backend.pipeline.expander import QueryExpander
from backend.pipeline.router import QueryRouter
from backend.schemas.result import RouterResult


@pytest.fixture
def expander():
    cfg = PipelineConfig(max_expanded_queries=5)
    return QueryExpander(cfg)


def _make_result(intent: str, n: int, strategy: str) -> RouterResult:
    return RouterResult(intent=intent, num_expansions=n, strategy=strategy)


def test_original_always_first(expander):
    query = "best GPU for AI"
    rr = _make_result("research", 5, "technical")
    results = expander.expand(query, rr)
    assert results[0] == query


def test_research_returns_multiple(expander):
    rr = _make_result("research", 5, "technical")
    results = expander.expand("best GPU for AI", rr)
    assert len(results) >= 2
    assert len(results) <= 6  # original + up to 5


def test_fact_returns_fewer(expander):
    rr = _make_result("fact", 2, "paraphrase")
    results = expander.expand("what is entropy", rr)
    assert len(results) <= 3


def test_compare_uses_comparison_terms(expander):
    rr = _make_result("compare", 4, "comparison")
    results = expander.expand("PyTorch vs TensorFlow", rr)
    joined = " ".join(results).lower()
    assert any(word in joined for word in ["vs", "alternatives", "comparison", "pros"])


def test_no_duplicates(expander):
    rr = _make_result("research", 5, "technical")
    results = expander.expand("machine learning tutorial", rr)
    assert len(results) == len(set(results))


def test_community_strategy(expander):
    rr = _make_result("troubleshoot", 3, "community")
    results = expander.expand("Python pip SSL error", rr)
    joined = " ".join(results).lower()
    assert any(word in joined for word in ["reddit", "forum", "community", "discussion"])


def test_cap_at_max_expanded(expander):
    rr = _make_result("research", 100, "technical")
    results = expander.expand("deep learning", rr)
    # max_expanded_queries is 5, so at most 6 total (original + 5)
    assert len(results) <= 6


def test_conversational_stays_small(expander):
    rr = _make_result("conversational", 1, "paraphrase")
    results = expander.expand("hello", rr)
    assert len(results) <= 2
