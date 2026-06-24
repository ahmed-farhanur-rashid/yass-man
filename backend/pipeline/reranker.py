"""
Reranker Stage — Phase 6.

Scores every result against the original query using a cross-encoder,
then returns the top-K sorted by relevance score.
"""

from __future__ import annotations

import asyncio
import logging

from backend.model_config_loader import RerankerConfig
from backend.models.reranker_model import RerankerModel
from backend.schemas.result import AggregatedResult, RankedResult

logger = logging.getLogger(__name__)


class RerankerStage:
    """
    Cross-encoder reranking pipeline stage.
    """

    def __init__(self, model: RerankerModel, config: RerankerConfig) -> None:
        self._model = model
        self._top_k = config.top_k

    async def rerank(
        self,
        query: str,
        results: list[AggregatedResult],
    ) -> list[RankedResult]:
        """
        Score and sort *results* by relevance to *query*.

        Returns up to ``config.top_k`` RankedResult objects in descending score order.
        Inference is CPU-bound so we wrap it in asyncio.to_thread().
        """
        if not results:
            return []

        documents = [f"{r.title}. {r.snippet}" for r in results]

        scores: list[float] = await asyncio.to_thread(
            self._model.score, query, documents
        )

        ranked_pairs = sorted(
            zip(results, scores), key=lambda pair: pair[1], reverse=True
        )

        top_k = ranked_pairs[: self._top_k]

        logger.info(
            "Reranker: %d → top-%d | best=%.4f worst=%.4f",
            len(results),
            len(top_k),
            top_k[0][1] if top_k else 0.0,
            top_k[-1][1] if top_k else 0.0,
        )
        for result, score in top_k:
            logger.debug("  %.4f  %s", score, result.url)

        return [
            RankedResult(
                title=r.title,
                snippet=r.snippet,
                url=r.url,
                score=score,
                query_sources=r.query_sources,
            )
            for r, score in top_k
        ]
