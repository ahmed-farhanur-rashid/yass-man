"""
Query Expansion Engine — Phase 2.

Turns one query into multiple search queries for better recall.
The original query is always the first item in the returned list.

Strategies:
  paraphrase    — rephrase using synonym substitution
  technical     — add technical qualifiers (docs, tutorial, guide)
  comparison    — "X vs alternatives", "best X 2026"
  community     — append "reddit", "forum" (surfaces real opinions)
  documentation — append "official docs", "how to"
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from backend.model_config_loader import PipelineConfig
from backend.schemas.result import RouterResult

logger = logging.getLogger(__name__)

# ── Synonym map for paraphrase expansion ─────────────────────────────────────
_SYNONYMS: dict[str, list[str]] = {
    "best": ["top", "recommended", "leading", "optimal"],
    "fast": ["quick", "rapid", "speedy", "high-performance"],
    "easy": ["simple", "beginner-friendly", "straightforward", "accessible"],
    "free": ["open-source", "no-cost", "gratis"],
    "how to": ["guide for", "steps to", "tutorial on"],
    "python": ["python3", "python programming"],
    "javascript": ["js", "node.js", "ecmascript"],
    "machine learning": ["ML", "deep learning", "AI/ML"],
    "artificial intelligence": ["AI", "machine learning"],
}

_TECHNICAL_QUALIFIERS = ["tutorial", "documentation", "guide", "explained", "examples"]
_COMMUNITY_QUALIFIERS = ["reddit", "forum", "discussion", "community experience"]
_DOC_QUALIFIERS = ["official documentation", "docs", "how to", "getting started"]
_CURRENT_YEAR = "2026"


class QueryExpander:
    """
    Generates expanded queries given an original query and router result.

    Instantiate once at startup and reuse across requests.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._max = config.max_expanded_queries

    def expand(self, query: str, router_result: RouterResult) -> list[str]:
        """
        Return a list of query strings.

        The first item is always the original query.
        Length is capped at ``pipeline.max_expanded_queries``.
        """
        intent = router_result.intent
        n = min(router_result.num_expansions, self._max)
        strategy = router_result.strategy

        variants: list[str] = [query]  # original always first

        # Pick expansion functions based on intent
        generators = _STRATEGY_GENERATORS.get(strategy, [_paraphrase_expand])

        for gen in generators:
            if len(variants) >= n + 1:
                break
            new = gen(query)
            for v in new:
                if v.strip().lower() != query.strip().lower() and v not in variants:
                    variants.append(v)
                if len(variants) >= n + 1:
                    break

        # Fallback: if we still need more, pull from other strategies
        fallback_gens = [g for g in _ALL_GENERATORS if g not in generators]
        for gen in fallback_gens:
            if len(variants) >= n + 1:
                break
            for v in gen(query):
                if v.strip().lower() != query.strip().lower() and v not in variants:
                    variants.append(v)
                if len(variants) >= n + 1:
                    break

        result = variants[: n + 1]
        logger.debug("Expander: query=%r → %d variants", query, len(result))
        for i, v in enumerate(result):
            logger.debug("  [%d] %s", i, v)
        return result


# ── Expansion strategy functions ──────────────────────────────────────────────


def _paraphrase_expand(query: str) -> list[str]:
    """Simple word substitution using the synonym map."""
    results: list[str] = []
    q_lower = query.lower()
    for word, synonyms in _SYNONYMS.items():
        if word in q_lower:
            for syn in synonyms[:2]:
                variant = re.sub(re.escape(word), syn, query, flags=re.IGNORECASE, count=1)
                if variant.lower() != query.lower():
                    results.append(variant)
    # If no synonym matched, try rephrasing structure
    if not results:
        results.append(f"{query} explained")
        results.append(f"what is {query}")
    return results


def _technical_expand(query: str) -> list[str]:
    """Add technical qualifiers."""
    return [f"{query} {qual}" for qual in _TECHNICAL_QUALIFIERS]


def _comparison_expand(query: str) -> list[str]:
    """Generate comparison and alternative queries."""
    return [
        f"{query} alternatives",
        f"best {query} {_CURRENT_YEAR}",
        f"{query} vs alternatives comparison",
        f"{query} pros and cons",
    ]


def _community_expand(query: str) -> list[str]:
    """Surface community discussion."""
    return [
        f"{query} reddit",
        f"{query} forum discussion",
        f"{query} community experience",
        f"{query} user reviews",
    ]


def _documentation_expand(query: str) -> list[str]:
    """Target official docs and getting-started guides."""
    return [
        f"{query} official documentation",
        f"{query} getting started",
        f"how to {query}",
        f"{query} docs",
    ]


_ALL_GENERATORS = [
    _paraphrase_expand,
    _technical_expand,
    _comparison_expand,
    _community_expand,
    _documentation_expand,
]

_STRATEGY_GENERATORS: dict[str, list] = {
    "paraphrase": [_paraphrase_expand, _technical_expand],
    "technical": [_technical_expand, _documentation_expand, _paraphrase_expand],
    "comparison": [_comparison_expand, _community_expand, _paraphrase_expand],
    "community": [_community_expand, _paraphrase_expand, _technical_expand],
    "documentation": [_documentation_expand, _technical_expand, _paraphrase_expand],
}
