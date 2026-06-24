"""
Aggregator — Phase 4.

Merges parallel retrieval results into a clean, deduplicated list.
Deduplication is URL-based (after normalization). Where the same URL
appears multiple times, snippets are merged.
"""

from __future__ import annotations

import logging

from backend.schemas.result import AggregatedResult, RawResult
from backend.utils.text_utils import merge_snippets
from backend.utils.url_utils import normalize_url

logger = logging.getLogger(__name__)


class Aggregator:
    """Stateless merge + URL-dedup stage. Safe to reuse across requests."""

    def aggregate(self, raw_results: list[RawResult]) -> list[AggregatedResult]:
        """
        Merge *raw_results* from all expanded queries into a deduplicated list.

        - Normalizes every URL before comparison
        - Keeps the first occurrence's title
        - Merges snippets (keeps longest, appends unique sentences from others)
        - Collects all query_sources and source_engines for provenance
        """
        # Map: normalized_url → accumulated data
        seen: dict[str, dict] = {}

        for r in raw_results:
            norm = normalize_url(r.url)

            if norm not in seen:
                seen[norm] = {
                    "title": r.title,
                    "url": r.url,  # keep the first canonical URL form
                    "normalized_url": norm,
                    "snippets": [r.snippet] if r.snippet else [],
                    "query_sources": [r.query_source] if r.query_source else [],
                    "source_engines": [r.source_engine] if r.source_engine else [],
                }
            else:
                entry = seen[norm]
                if r.snippet:
                    entry["snippets"].append(r.snippet)
                if r.query_source and r.query_source not in entry["query_sources"]:
                    entry["query_sources"].append(r.query_source)
                if r.source_engine and r.source_engine not in entry["source_engines"]:
                    entry["source_engines"].append(r.source_engine)

        results: list[AggregatedResult] = []
        for data in seen.values():
            merged_snippet = merge_snippets(data["snippets"]) if data["snippets"] else ""
            results.append(
                AggregatedResult(
                    title=data["title"],
                    snippet=merged_snippet,
                    url=data["url"],
                    normalized_url=data["normalized_url"],
                    query_sources=data["query_sources"],
                    source_engines=data["source_engines"],
                )
            )

        logger.info(
            "Aggregator: %d raw → %d deduplicated results",
            len(raw_results),
            len(results),
        )
        return results
