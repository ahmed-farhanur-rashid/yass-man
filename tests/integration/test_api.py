"""
HTTP-level integration tests for the FastAPI app.

Uses TestClient (synchronous HTTPX wrapper) with the full app mounted.
All ML models and SearXNG are mocked — no network or GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── App setup with mocked models ──────────────────────────────────────────────
#
# We patch the model-loading callables before importing the app so the lifespan
# never tries to hit disk or HuggingFace.

def _make_mock_embedder():
    m = MagicMock()
    # Return an identity matrix sized to the actual input — never out of bounds.
    m.embed.side_effect = lambda texts: np.eye(len(texts), dtype=np.float32)
    return m

def _make_mock_reranker():
    m = MagicMock()
    m.score.side_effect = lambda q, docs: [0.9 - i * 0.05 for i in range(len(docs))]
    return m

def _make_fake_searxng_response(query: str) -> dict:
    return {
        "results": [
            {
                "title": f"Result {i} for {query}",
                "url": f"https://result{i}.example.com/page",
                "content": f"This is a useful snippet about {query} number {i}.",
                "engine": "google",
            }
            for i in range(5)
        ]
    }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("logs")

    # Patch model constructors so lifespan doesn't load real weights
    with (
        patch("backend.models.embedder_model.EmbedderModel", return_value=_make_mock_embedder()),
        patch("backend.models.reranker_model.RerankerModel", return_value=_make_mock_reranker()),
        patch("backend.config.get_settings") as mock_settings,
    ):
        from backend.config import Settings
        s = Settings(
            searxng_url="https://fake-searxng.test",
            log_dir=tmp,
            enable_llm=False,
            enable_clustering=True,
            enable_feedback=True,
        )
        mock_settings.return_value = s

        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            # Inject mocked pipeline components into app state
            app.state.embedder_model = _make_mock_embedder()
            app.state.reranker_model = _make_mock_reranker()
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200

def test_health_response_schema(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "models_loaded" in data
    assert "searxng_reachable" in data


# ── /search ───────────────────────────────────────────────────────────────────

def test_search_requires_q_param(client):
    resp = client.get("/search")
    assert resp.status_code == 422  # FastAPI validation error

def test_search_q_too_short(client):
    resp = client.get("/search", params={"q": ""})
    assert resp.status_code == 422

def test_search_q_too_long(client):
    resp = client.get("/search", params={"q": "x" * 501})
    assert resp.status_code == 422

def test_search_returns_200_with_mocked_searxng(client):
    with patch("backend.pipeline.retriever.Retriever._do_fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = []  # SearXNG returns nothing — pipeline degrades gracefully
        resp = client.get("/search", params={"q": "test query"})
    assert resp.status_code == 200

def test_search_response_schema(client):
    with patch("backend.pipeline.retriever.Retriever._do_fetch", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = []
        data = client.get("/search", params={"q": "entropy"}).json()
    assert "query" in data
    assert "results" in data
    assert "expanded_queries" in data
    assert "latency_ms" in data
    assert data["query"] == "entropy"

def test_search_with_results(client):
    from backend.schemas.result import RawResult

    fake_raw = [
        RawResult(
            title=f"Result {i}",
            url=f"https://site{i}.com/",
            snippet=f"Relevant content about the topic number {i}.",
            source_engine="google",
            query_source="query",
        )
        for i in range(5)
    ]

    with patch("backend.pipeline.retriever.Retriever.retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = fake_raw
        resp = client.get("/search", params={"q": "machine learning"})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) > 0
    result = data["results"][0]
    assert "title" in result
    assert "url" in result
    assert "snippet" in result
    assert "score" in result

def test_search_top_k_respected(client):
    from backend.schemas.result import RawResult

    fake_raw = [
        RawResult(
            title=f"R{i}", url=f"https://s{i}.com/",
            snippet=f"snippet {i} " * 10, source_engine="g", query_source="q"
        )
        for i in range(20)
    ]
    with patch("backend.pipeline.retriever.Retriever.retrieve", new_callable=AsyncMock) as mock:
        mock.return_value = fake_raw
        resp = client.get("/search", params={"q": "deep learning", "top_k": 3})

    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= 3

def test_search_is_cached_on_second_call(client):
    """Second identical query should be served from cache (no retriever call)."""
    from backend.schemas.result import RawResult

    fake_raw = [
        RawResult(title="Cached", url="https://cached.com/", snippet="cache test content here", source_engine="g", query_source="q")
    ]
    call_count = 0

    async def counting_retrieve(queries):
        nonlocal call_count
        call_count += 1
        return fake_raw

    # Clear the result cache before this test
    client.app.state.result_cache.clear()

    with patch("backend.pipeline.retriever.Retriever.retrieve", side_effect=counting_retrieve):
        client.get("/search", params={"q": "cache test query xyz"})
        client.get("/search", params={"q": "cache test query xyz"})

    assert call_count == 1  # second call served from cache

def test_search_latency_breakdown_present(client):
    with patch("backend.pipeline.retriever.Retriever._do_fetch", new_callable=AsyncMock) as mock:
        mock.return_value = []
        data = client.get("/search", params={"q": "latency test"}).json()
    lb = data["latency_ms"]
    assert "total" in lb
    assert lb["total"] >= 0


# ── /feedback ─────────────────────────────────────────────────────────────────

def test_feedback_up(client):
    resp = client.post("/feedback", json={
        "query_id": "test-uuid",
        "result_url": "https://example.com/",
        "signal": "up",
    })
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_feedback_down(client):
    resp = client.post("/feedback", json={
        "query_id": "test-uuid",
        "result_url": "https://example.com/",
        "signal": "down",
    })
    assert resp.status_code == 200

def test_feedback_invalid_signal(client):
    resp = client.post("/feedback", json={
        "query_id": "test-uuid",
        "result_url": "https://example.com/",
        "signal": "meh",  # not "up" or "down"
    })
    assert resp.status_code == 422

def test_feedback_missing_fields(client):
    resp = client.post("/feedback", json={"signal": "up"})
    assert resp.status_code == 422


# ── /click ────────────────────────────────────────────────────────────────────

def test_click_logged(client):
    resp = client.post("/click", json={
        "query_id": "test-uuid",
        "result_url": "https://clicked.com/",
    })
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_click_missing_fields(client):
    resp = client.post("/click", json={"query_id": "only-id"})
    assert resp.status_code == 422
