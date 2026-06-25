"""
Synthesizer — Phase 7.

Generates a cited answer from top-K results using the local GGUF LLM.
If synthesis fails or times out, returns None — the pipeline continues gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from backend.model_config_loader import LLMConfig
from backend.models.llm_model import LLMModel
from backend.schemas.result import Citation, RankedResult, SynthesisResult
from backend.utils.text_utils import clean_snippet

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_SYSTEM_PROMPT = """\
You are a precise search assistant. Answer the user's query using ONLY the information in the sources below.
Cite sources inline as [1], [2], etc. immediately after each claim.
If sources contradict each other, note the conflict explicitly with "Note: [source A] says X while [source B] says Y."
Do not include information not present in the sources.
Do not add preamble. Start your answer directly.
Keep the answer concise (3-5 sentences unless the query demands more).\
"""


class Synthesizer:
    """
    LLM-based answer synthesis with inline citations.
    """

    def __init__(self, model: LLMModel, config: LLMConfig) -> None:
        self._model = model
        self._config = config

    async def synthesize(
        self,
        query: str,
        results: list[RankedResult],
    ) -> Optional[SynthesisResult]:
        """
        Generate a cited answer from *results*.

        Returns None if synthesis fails, times out, or produces no useful output.
        The pipeline continues without an answer in that case.
        """
        if not results:
            return None

        top = results[: self._config.max_sources_in_prompt]
        prompt = self._build_prompt(query, top)

        try:
            raw_answer: str = await asyncio.to_thread(
                self._model.generate,
                prompt,
                self._config.max_answer_tokens,
            )
        except Exception as exc:
            logger.error("Synthesizer: generation failed: %s", exc)
            return None

        if not raw_answer or len(raw_answer.strip()) < 20:
            logger.warning("Synthesizer: answer too short, skipping")
            return None

        citations = self._extract_citations(raw_answer, top)

        # Require at least one citation if configured
        if self._config.require_citations and not citations:
            logger.warning("Synthesizer: no citations found in answer, skipping")
            return None

        logger.info(
            "Synthesizer: generated %d-char answer with %d citations",
            len(raw_answer),
            len(citations),
        )
        return SynthesisResult(answer=raw_answer, citations=citations)

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, results: list[RankedResult]) -> str:
        sources_block = "\n\n".join(
            f"[{i + 1}] {r.title}\n{clean_snippet(r.snippet, 400)}"
            for i, r in enumerate(results)
        )
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Query: {query}\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"Answer:"
        )

    def _extract_citations(
        self,
        answer: str,
        results: list[RankedResult],
    ) -> list[Citation]:
        """Parse [N] markers from the answer and map them to result URLs."""
        found_indices: list[int] = []
        seen: set[int] = set()

        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            if idx not in seen and 1 <= idx <= len(results):
                seen.add(idx)
                found_indices.append(idx)

        return [
            Citation(
                index=idx,
                title=results[idx - 1].title,
                url=results[idx - 1].url,
            )
            for idx in found_indices
        ]
