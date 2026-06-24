"""
Query Router — Phase 1.

Classifies an incoming query into an intent type, which controls how the
downstream pipeline behaves (how many expansions, which strategies to use).

Intent types:
  fact          — short factual lookups, "what is X", "who is Y"
  compare       — "X vs Y", "difference between A and B"
  troubleshoot  — error messages, "not working", "how to fix"
  research      — tutorials, guides, "best X", "how to"
  conversational — greetings, single words, chitchat
"""

from __future__ import annotations

import logging
import re

from backend.model_config_loader import RouterConfig
from backend.schemas.result import RouterResult

logger = logging.getLogger(__name__)

# ── Intent → number of expanded queries ───────────────────────────────────────
_EXPANSION_COUNT: dict[str, int] = {
    "fact": 2,
    "compare": 4,
    "troubleshoot": 3,
    "research": 5,
    "conversational": 1,
}

# ── Intent → preferred expansion strategy ────────────────────────────────────
_STRATEGY: dict[str, str] = {
    "fact": "paraphrase",
    "compare": "comparison",
    "troubleshoot": "community",
    "research": "technical",
    "conversational": "paraphrase",
}

# ── Rule patterns (compiled once at import time) ──────────────────────────────
_COMPARE_PATTERNS = re.compile(
    r"\bvs\.?\b|\bversus\b|difference between|better than|compared? to"
    r"|\bor\b.{1,30}\bor\b|\bpros? and cons?\b|\bwhich is better\b",
    re.IGNORECASE,
)

_TROUBLESHOOT_PATTERNS = re.compile(
    r"not working|doesn'?t? work|won'?t work|error\b|fix\b|broken\b"
    r"|why is.{1,30}(not|slow|fail|crash)|how to fix|can'?t\b|cannot\b"
    r"|exception\b|traceback\b|bug\b|issue\b|problem\b",
    re.IGNORECASE,
)

_FACT_PATTERNS = re.compile(
    r"^what (is|are|was|were)\b|^who (is|was|are|were)\b|^when (did|was|is)\b"
    r"|^where (is|was|are)\b|^define\b|^meaning of\b|^how (much|many|old|tall|far|long|big)\b",
    re.IGNORECASE,
)

_RESEARCH_PATTERNS = re.compile(
    r"\bbest\b|\bhow to\b|\bguide\b|\btutorial\b|\bexplain\b|\boverview\b"
    r"|\breview\b|\brecommend\b|\blearn\b|\bintro(duction)?\b|\bbeginners?\b"
    r"|\bcourse\b|\bexample\b|\bsetup\b|\binstall\b|\bgetting started\b",
    re.IGNORECASE,
)

_CONVERSATIONAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|sup|yo|thanks?|thank you|ok|okay|sure|lol|haha|hmm)[\s!?.]*$",
    re.IGNORECASE,
)


class QueryRouter:
    """
    Classifies query intent and returns a RouterResult.

    Instantiate once at startup and reuse across requests.
    """

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        if config.mode == "model":
            logger.warning(
                "router.mode='model' is set in model_config.yaml, but "
                "model-based routing is not yet implemented. Using rule-based fallback."
            )

    def route(self, query: str) -> RouterResult:
        """Classify *query* and return a RouterResult."""
        intent = self._classify(query.strip())
        result = RouterResult(
            intent=intent,
            num_expansions=_EXPANSION_COUNT[intent],
            strategy=_STRATEGY[intent],
        )
        logger.debug("Router: query=%r → intent=%s expansions=%d", query, intent, result.num_expansions)
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify(self, query: str) -> str:
        # Single word or greeting → conversational
        if _CONVERSATIONAL_PATTERNS.match(query):
            return "conversational"

        words = query.split()
        if len(words) <= 1:
            return "conversational"

        # Comparison queries
        if _COMPARE_PATTERNS.search(query):
            return "compare"

        # Troubleshooting queries
        if _TROUBLESHOOT_PATTERNS.search(query):
            return "troubleshoot"

        # Short factual queries (< 4 words) OR explicit fact patterns
        if _FACT_PATTERNS.match(query) or (len(words) < 4 and not _RESEARCH_PATTERNS.search(query)):
            return "fact"

        # Research / how-to
        if _RESEARCH_PATTERNS.search(query):
            return "research"

        # Default: research (better recall than fact for ambiguous queries)
        return "research"
