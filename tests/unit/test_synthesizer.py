"""Unit tests for the Synthesizer pipeline stage (LLM is mocked)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.model_config_loader import LLMConfig
from backend.pipeline.synthesizer import Synthesizer
from backend.schemas.result import RankedResult


def _make_config(require_citations: bool = True) -> LLMConfig:
    return LLMConfig(
        enabled=True,
        model="mock",
        path="./models/mock.gguf",
        download_url=None,
        max_sources_in_prompt=8,
        require_citations=require_citations,
    )


def _make_result(n: int, url: str = None, title: str = None) -> RankedResult:
    return RankedResult(
        title=title or f"Result {n}",
        snippet=f"This is the snippet for result {n} with relevant content.",
        url=url or f"https://site{n}.com/page",
        score=1.0 - n * 0.1,
    )


def _make_synth(answer: str, require_citations: bool = True) -> Synthesizer:
    model = MagicMock()
    model.generate.return_value = answer
    return Synthesizer(model, _make_config(require_citations))


# ── synthesize ────────────────────────────────────────────────────────────────

def test_returns_none_for_empty_results():
    synth = _make_synth("anything")
    result = asyncio.run(synth.synthesize("query", []))
    assert result is None


def test_returns_none_when_answer_too_short():
    synth = _make_synth("ok")  # < 20 chars
    result = asyncio.run(synth.synthesize("query", [_make_result(1)]))
    assert result is None


def test_returns_none_when_generation_raises():
    model = MagicMock()
    model.generate.side_effect = RuntimeError("model crashed")
    synth = Synthesizer(model, _make_config())
    result = asyncio.run(synth.synthesize("query", [_make_result(1)]))
    assert result is None


def test_returns_none_when_citations_required_but_absent():
    # require_citations=True, but LLM returns answer with no [N] markers
    synth = _make_synth("The answer is forty-two and that is the truth.", require_citations=True)
    result = asyncio.run(synth.synthesize("query", [_make_result(1)]))
    assert result is None


def test_returns_result_when_citations_not_required_and_absent():
    synth = _make_synth(
        "The answer is forty-two and that is the truth.",
        require_citations=False,
    )
    result = asyncio.run(synth.synthesize("query", [_make_result(1)]))
    assert result is not None
    assert result.answer.startswith("The answer")
    assert result.citations == []


def test_extracts_citation_indices_correctly():
    synth = _make_synth("First claim [1]. Second claim [2]. Third from first again [1].")
    results = [_make_result(i, url=f"https://s{i}.com/") for i in range(1, 4)]
    result = asyncio.run(synth.synthesize("query", results))
    assert result is not None
    indices = [c.index for c in result.citations]
    # [1] appears twice but should deduplicate
    assert indices == [1, 2]


def test_citation_maps_to_correct_url():
    r1 = _make_result(1, url="https://alpha.com/", title="Alpha")
    r2 = _make_result(2, url="https://beta.com/", title="Beta")
    synth = _make_synth("Alpha says [1]. Beta says [2].")
    result = asyncio.run(synth.synthesize("query", [r1, r2]))
    assert result is not None
    assert result.citations[0].url == "https://alpha.com/"
    assert result.citations[1].url == "https://beta.com/"


def test_out_of_range_citation_ignored():
    synth = _make_synth("Valid [1]. Way out of range [99].")
    result = asyncio.run(synth.synthesize("query", [_make_result(1)]))
    assert result is not None
    assert len(result.citations) == 1
    assert result.citations[0].index == 1


def test_respects_max_sources_in_prompt():
    model = MagicMock()
    model.generate.return_value = "Answer using [1]."
    config = LLMConfig(
        enabled=True, model="m", path="p", download_url=None,
        max_sources_in_prompt=3, require_citations=True,
    )
    synth = Synthesizer(model, config)
    results = [_make_result(i) for i in range(1, 10)]
    asyncio.run(synth.synthesize("query", results))

    # Inspect the prompt passed to the model
    call_args = model.generate.call_args
    prompt = call_args[0][0]
    # Only sources [1], [2], [3] should appear
    assert "[4]" not in prompt
    assert "[3]" in prompt


# ── _build_prompt ─────────────────────────────────────────────────────────────

def test_prompt_contains_query():
    synth = _make_synth("irrelevant")
    results = [_make_result(1)]
    # Access private method directly to inspect prompt content
    prompt = synth._build_prompt("what is entropy", results)
    assert "what is entropy" in prompt


def test_prompt_contains_source_titles():
    synth = _make_synth("irrelevant")
    r = _make_result(1, title="Entropy Explained")
    prompt = synth._build_prompt("entropy", [r])
    assert "Entropy Explained" in prompt
