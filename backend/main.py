"""
YASS-MAN — FastAPI Application Entrypoint.

Lifespan loads all models once. Routes call pipeline stages in order:
router → expander → retriever → aggregator → embedder → reranker → synthesizer → logger
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from backend.cache.embedding_cache import EmbeddingCache
from backend.cache.result_cache import ResultCache
from backend.config import Settings, get_settings
from backend.logging.feedback import FeedbackLogger
from backend.logging.query_logger import QueryLogger
from backend.model_config_loader import ModelConfig, load_model_config
from backend.models.embedder_model import EmbedderModel
from backend.models.llm_model import LLMModel
from backend.models.reranker_model import RerankerModel
from backend.pipeline.aggregator import Aggregator
from backend.pipeline.embedder import EmbedderStage
from backend.pipeline.expander import QueryExpander
from backend.pipeline.reranker import RerankerStage
from backend.pipeline.retriever import Retriever
from backend.pipeline.router import QueryRouter
from backend.pipeline.synthesizer import Synthesizer
from backend.schemas.result import (
    ClickRequest,
    FeedbackRequest,
    HealthResponse,
    LatencyBreakdown,
    SearchResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dependency helpers (inlined — all are single app.state lookups) ───────────

def _get_router(req: Request) -> QueryRouter:
    return req.app.state.query_router

def _get_expander(req: Request) -> QueryExpander:
    return req.app.state.query_expander

def _get_retriever(req: Request) -> Retriever:
    return req.app.state.retriever

def _get_aggregator(req: Request) -> Aggregator:
    return req.app.state.aggregator

def _get_embedder_stage(req: Request) -> EmbedderStage:
    return req.app.state.embedder_stage

def _get_reranker_stage(req: Request) -> RerankerStage:
    return req.app.state.reranker_stage

def _get_synthesizer(req: Request) -> Optional[Synthesizer]:
    return getattr(req.app.state, "synthesizer", None)

def _get_query_logger(req: Request) -> QueryLogger:
    return req.app.state.query_logger

def _get_feedback_logger(req: Request) -> FeedbackLogger:
    return req.app.state.feedback_logger


def _get_result_cache(req: Request) -> ResultCache:
    return req.app.state.result_cache


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    model_cfg = load_model_config(str(settings.model_config_path))

    app.state.settings = settings
    app.state.model_cfg = model_cfg

    # ── HTTP client (shared across all requests) ──────────────────────────────
    http_client = httpx.AsyncClient(
        headers={"Accept": "application/json"},
        follow_redirects=True,
    )
    app.state.http_client = http_client

    # ── Load ML models ────────────────────────────────────────────────────────
    logger.info("Loading embedder…")
    embedder_model = EmbedderModel(model_cfg.embedder)
    app.state.embedder_model = embedder_model

    logger.info("Loading reranker…")
    reranker_model = RerankerModel(model_cfg.reranker)
    app.state.reranker_model = reranker_model

    llm_model: Optional[LLMModel] = None
    if model_cfg.llm.enabled and settings.enable_llm:
        logger.info("Loading LLM…")
        try:
            llm_model = LLMModel(model_cfg.llm)
        except FileNotFoundError as exc:
            logger.warning("LLM not loaded (model file missing): %s", exc)
        except Exception as exc:
            logger.warning("LLM not loaded (error): %s", exc)
    else:
        logger.info("LLM disabled (enable_llm=false or llm.enabled=false)")

    app.state.llm_model = llm_model

    # ── Assemble pipeline components ──────────────────────────────────────────
    embedding_cache = EmbeddingCache()
    app.state.embedding_cache = embedding_cache

    result_cache = ResultCache()
    app.state.result_cache = result_cache

    app.state.query_router = QueryRouter(model_cfg.router)
    app.state.query_expander = QueryExpander(model_cfg.pipeline)
    app.state.retriever = Retriever(settings, model_cfg.pipeline, http_client)
    app.state.aggregator = Aggregator()
    app.state.embedder_stage = EmbedderStage(
        model=embedder_model,
        cache=embedding_cache,
        config=model_cfg.pipeline,
        enabled=settings.enable_clustering,
    )
    app.state.reranker_stage = RerankerStage(reranker_model, model_cfg.reranker)
    app.state.synthesizer = (
        Synthesizer(llm_model, model_cfg.llm) if llm_model else None
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    app.state.query_logger = QueryLogger(settings.log_dir)
    app.state.feedback_logger = FeedbackLogger(settings.log_dir)

    # Mount static frontend LAST so API routes always take priority
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info("Frontend mounted from: %s", frontend_dir)
    else:
        logger.warning("Frontend dir not found at %s — UI will not be served", frontend_dir)

    logger.info("\u2713 YASS-MAN ready at http://%s:%d", settings.host, settings.port)

    yield

    # ── Cleanup ───────────────────────────────────────────────────────────────
    await http_client.aclose()
    logger.info("YASS-MAN shut down.")


# ── App ───────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="YASS-MAN",
    description="Yet Another SearXNG Search Meta AI Network — privacy-respecting AI-enhanced search",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    request_top_k: int = Query(default=10, alias="top_k", ge=1, le=50),
    router: Annotated[QueryRouter, Depends(_get_router)] = None,
    expander: Annotated[QueryExpander, Depends(_get_expander)] = None,
    retriever: Annotated[Retriever, Depends(_get_retriever)] = None,
    aggregator: Annotated[Aggregator, Depends(_get_aggregator)] = None,
    embedder_stage: Annotated[EmbedderStage, Depends(_get_embedder_stage)] = None,
    reranker_stage: Annotated[RerankerStage, Depends(_get_reranker_stage)] = None,
    synthesizer: Annotated[Optional[Synthesizer], Depends(_get_synthesizer)] = None,
    query_logger: Annotated[QueryLogger, Depends(_get_query_logger)] = None,
    result_cache: Annotated[ResultCache, Depends(_get_result_cache)] = None,
) -> SearchResponse:
    # ── Cache lookup ──────────────────────────────────────────────────────────
    cached = result_cache.get(q)
    if cached is not None:
        return cached

    query_id = str(uuid.uuid4())
    t_total_start = time.perf_counter()
    latency: dict[str, float] = {}

    # ── 1. Router ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    router_result = router.route(q)
    latency["router"] = _ms(t0)

    # ── 2. Expander ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    expanded_queries = expander.expand(q, router_result)
    latency["expansion"] = _ms(t0)

    # ── 3. Retriever ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    raw_results = await retriever.retrieve(expanded_queries)
    latency["search"] = _ms(t0)
    num_retrieved = len(raw_results)

    # ── 4. Aggregator ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    aggregated = aggregator.aggregate(raw_results)
    latency["aggregation"] = _ms(t0)
    num_after_dedup = len(aggregated)

    # ── 5. Embedding + Clustering ─────────────────────────────────────────────
    t0 = time.perf_counter()
    clustered = await embedder_stage.cluster(aggregated)
    latency["embedding"] = _ms(t0)
    num_after_clustering = len(clustered)

    # ── 6. Reranker ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    ranked = await reranker_stage.rerank(q, clustered)
    latency["rerank"] = _ms(t0)

    # ── 7. Synthesizer ────────────────────────────────────────────────────────
    synthesis = None
    t0 = time.perf_counter()
    if synthesizer and ranked:
        synthesis = await synthesizer.synthesize(q, ranked)
    latency["synthesis"] = _ms(t0)

    latency["total"] = _ms(t_total_start)

    # ── 8. Logger ─────────────────────────────────────────────────────────────
    query_logger.log(
        query_id=query_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=q,
        expanded_queries=expanded_queries,
        num_results_retrieved=num_retrieved,
        num_results_after_dedup=num_after_dedup,
        num_results_after_clustering=num_after_clustering,
        top_k_urls=[r.url for r in ranked],
        reranker_scores=[r.score for r in ranked],
        answer_generated=synthesis is not None,
        latency_ms=latency,
    )

    response = SearchResponse(
        query_id=query_id,
        query=q,
        expanded_queries=expanded_queries,
        answer=synthesis.answer if synthesis else None,
        citations=synthesis.citations if synthesis else [],
        results=ranked[:request_top_k],
        latency_ms=LatencyBreakdown(**latency),
    )
    result_cache.set(q, response)
    return response


@app.post("/feedback", status_code=200)
async def feedback(
    body: FeedbackRequest,
    feedback_logger: Annotated[FeedbackLogger, Depends(_get_feedback_logger)] = None,
) -> dict:
    if not app.state.settings.enable_feedback:
        raise HTTPException(status_code=404, detail="Feedback is disabled")
    feedback_logger.log_feedback(
        query_id=body.query_id,
        result_url=body.result_url,
        signal=body.signal,
    )
    return {"status": "ok"}


@app.post("/click", status_code=200)
async def click(
    body: ClickRequest,
    feedback_logger: Annotated[FeedbackLogger, Depends(_get_feedback_logger)] = None,
) -> dict:
    feedback_logger.log_click(
        query_id=body.query_id,
        result_url=body.result_url,
    )
    return {"status": "ok"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = app.state
    settings: Settings = state.settings
    model_cfg: ModelConfig = state.model_cfg

    # Ping SearXNG
    searxng_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(
                settings.searxng_search_url,
                params={"q": "test", "format": "json"},
            )
            searxng_ok = resp.status_code == 200
    except Exception:
        pass

    models_loaded = (
        getattr(state, "embedder_model", None) is not None
        and getattr(state, "reranker_model", None) is not None
    )

    llm_model = getattr(state, "llm_model", None)

    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
        searxng_reachable=searxng_ok,
        embedder=model_cfg.embedder.model if models_loaded else None,
        reranker=model_cfg.reranker.model if models_loaded else None,
        llm=model_cfg.llm.model if llm_model else None,
        llm_enabled=llm_model is not None,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ms(t_start: float) -> float:
    """Return elapsed milliseconds since t_start."""
    return round((time.perf_counter() - t_start) * 1000, 1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("backend.main:app", host=s.host, port=s.port, reload=s.reload)
