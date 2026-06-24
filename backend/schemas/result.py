"""Pydantic models for search pipeline results and API responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Raw pipeline data ─────────────────────────────────────────────────────────


class RawResult(BaseModel):
    """A single result returned by SearXNG for one query."""

    title: str
    snippet: str
    url: str
    source_engine: str = ""
    query_source: str = ""  # which expanded query surfaced this


class AggregatedResult(BaseModel):
    """After URL deduplication and snippet merging."""

    title: str
    snippet: str
    url: str
    normalized_url: str
    query_sources: list[str] = Field(default_factory=list)
    source_engines: list[str] = Field(default_factory=list)


class RankedResult(BaseModel):
    """After embedding clustering and cross-encoder reranking."""

    title: str
    snippet: str
    url: str
    score: float
    query_sources: list[str] = Field(default_factory=list)


# ── Synthesis ─────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    index: int
    title: str
    url: str


class SynthesisResult(BaseModel):
    answer: str
    citations: list[Citation]


# ── Router ────────────────────────────────────────────────────────────────────


class RouterResult(BaseModel):
    intent: str  # fact | compare | troubleshoot | research | conversational
    num_expansions: int
    strategy: str


# ── API response ──────────────────────────────────────────────────────────────


class LatencyBreakdown(BaseModel):
    router: float = 0.0
    expansion: float = 0.0
    search: float = 0.0
    aggregation: float = 0.0
    embedding: float = 0.0
    rerank: float = 0.0
    synthesis: float = 0.0
    total: float = 0.0


class SearchResponse(BaseModel):
    query_id: str
    query: str
    expanded_queries: list[str] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    results: list[RankedResult] = Field(default_factory=list)
    latency_ms: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class FeedbackRequest(BaseModel):
    query_id: str
    result_url: str
    signal: str = Field(..., pattern="^(up|down)$")


class ClickRequest(BaseModel):
    query_id: str
    result_url: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    searxng_reachable: bool
    embedder: Optional[str] = None
    reranker: Optional[str] = None
    llm: Optional[str] = None
    llm_enabled: bool = False
