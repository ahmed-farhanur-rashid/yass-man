"""Unit tests for the query router."""

import pytest
from backend.model_config_loader import RouterConfig
from backend.pipeline.router import QueryRouter


@pytest.fixture
def router():
    return QueryRouter(RouterConfig(mode="rule-based"))


# ── Fact ─────────────────────────────────────────────────────────────────────
def test_fact_what_is(router):
    r = router.route("what is quantum entanglement")
    assert r.intent == "fact"
    assert r.num_expansions == 2


def test_fact_who_is(router):
    r = router.route("who is the CEO of Anthropic")
    assert r.intent == "fact"


def test_fact_short_query(router):
    r = router.route("DNA structure")
    assert r.intent == "fact"


def test_fact_define(router):
    r = router.route("define photosynthesis")
    assert r.intent == "fact"


# ── Compare ───────────────────────────────────────────────────────────────────
def test_compare_vs(router):
    r = router.route("Python vs JavaScript performance")
    assert r.intent == "compare"
    assert r.num_expansions == 4


def test_compare_versus(router):
    r = router.route("Mac versus Windows for development")
    assert r.intent == "compare"


def test_compare_difference(router):
    r = router.route("difference between REST and GraphQL")
    assert r.intent == "compare"


def test_compare_better_than(router):
    r = router.route("is React better than Vue")
    assert r.intent == "compare"


# ── Troubleshoot ──────────────────────────────────────────────────────────────
def test_troubleshoot_not_working(router):
    r = router.route("Python pip not working on Windows")
    assert r.intent == "troubleshoot"
    assert r.num_expansions == 3


def test_troubleshoot_error(router):
    r = router.route("ModuleNotFoundError numpy error")
    assert r.intent == "troubleshoot"


def test_troubleshoot_how_to_fix(router):
    r = router.route("how to fix CORS error in FastAPI")
    assert r.intent == "troubleshoot"


# ── Research ──────────────────────────────────────────────────────────────────
def test_research_best(router):
    r = router.route("best GPU for machine learning 2026")
    assert r.intent == "research"
    assert r.num_expansions == 5


def test_research_how_to(router):
    r = router.route("how to deploy a FastAPI app to AWS")
    assert r.intent == "research"


def test_research_tutorial(router):
    r = router.route("Rust programming tutorial for beginners")
    assert r.intent == "research"


# ── Conversational ────────────────────────────────────────────────────────────
def test_conversational_greeting(router):
    r = router.route("hello")
    assert r.intent == "conversational"
    assert r.num_expansions == 1


def test_conversational_thanks(router):
    r = router.route("thanks!")
    assert r.intent == "conversational"


def test_conversational_single_word(router):
    r = router.route("python")
    assert r.intent == "conversational"


# ── Strategy mapping ──────────────────────────────────────────────────────────
def test_strategy_assigned(router):
    assert router.route("what is entropy").strategy == "paraphrase"
    assert router.route("PyTorch vs TensorFlow").strategy == "comparison"
    assert router.route("pip install error SSL").strategy == "community"
    assert router.route("best databases for time series").strategy == "technical"
