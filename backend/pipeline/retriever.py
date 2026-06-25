"""
Parallel Retriever — Phase 3.

Fires all expanded queries at SearXNG concurrently using asyncio.gather().
Handles timeouts and HTTP errors gracefully — partial results are fine, crashes are not.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from backend.config import Settings
from backend.model_config_loader import PipelineConfig
from backend.schemas.result import RawResult
from backend.utils.text_utils import clean_snippet

logger = logging.getLogger(__name__)

class Retriever:
    """
    Async SearXNG interface.

    Uses a shared ``httpx.AsyncClient`` that is created once at startup
    (passed in) rather than per-request.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline_config: PipelineConfig,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._base_url = settings.searxng_search_url
        self._timeout = pipeline_config.search_timeout_seconds
        self._client = http_client

    async def retrieve(self, queries: list[str]) -> list[RawResult]:
        """
        Execute all *queries* in parallel against SearXNG.

        Returns the combined list of raw results (not deduplicated).
        Any query that times out or errors contributes an empty list.
        """
        tasks = [self._fetch_one(q) for q in queries]
        per_query_results = await asyncio.gather(*tasks, return_exceptions=False)

        all_results: list[RawResult] = []
        for results in per_query_results:
            all_results.extend(results)

        logger.info(
            "Retriever: %d queries → %d total raw results",
            len(queries),
            len(all_results),
        )
        return all_results

    # ── Private ───────────────────────────────────────────────────────────────

    async def _fetch_one(self, query: str) -> list[RawResult]:
        """Fetch results for a single query; retry once on failure."""
        for attempt in range(2):
            try:
                return await self._do_fetch(query)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Retriever: query=%r failed (%s), retrying…", query, exc
                    )
                else:
                    logger.warning(
                        "Retriever: query=%r failed after retry (%s), skipping", query, exc
                    )
            except Exception as exc:
                logger.error("Retriever: unexpected error for query=%r: %s", query, exc)
                break
        return []

    async def _do_fetch(self, query: str) -> list[RawResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": "0",
        }
        resp = await self._client.get(
            self._base_url,
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_results: list[RawResult] = []
        for item in data.get("results", []):
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            snippet = clean_snippet(item.get("content", ""))
            engine = item.get("engine", "") or ",".join(item.get("engines", []))

            if not url:
                continue

            raw_results.append(
                RawResult(
                    title=title or url,
                    snippet=snippet,
                    url=url,
                    source_engine=engine,
                    query_source=query,
                )
            )

        logger.debug("Retriever: query=%r → %d results", query, len(raw_results))
        return raw_results
