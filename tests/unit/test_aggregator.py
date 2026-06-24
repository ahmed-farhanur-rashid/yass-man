"""Unit tests for the aggregator and URL utilities."""

import pytest
from backend.pipeline.aggregator import Aggregator
from backend.schemas.result import RawResult
from backend.utils.url_utils import extract_domain, normalize_url


# ── URL utils ─────────────────────────────────────────────────────────────────

def test_normalize_strips_utm():
    url = "https://example.com/page?utm_source=google&utm_medium=cpc&q=test"
    norm = normalize_url(url)
    assert "utm_source" not in norm
    assert "utm_medium" not in norm
    assert "q=test" in norm


def test_normalize_strips_www():
    assert normalize_url("https://www.example.com/") == normalize_url("https://example.com/")


def test_normalize_trailing_slash():
    a = normalize_url("https://example.com/page/")
    b = normalize_url("https://example.com/page")
    assert a == b


def test_normalize_lowercase_domain():
    norm = normalize_url("https://EXAMPLE.COM/Page")
    assert "example.com" in norm


def test_normalize_strips_fbclid():
    url = "https://example.com/post?fbclid=abc123"
    assert "fbclid" not in normalize_url(url)


def test_extract_domain():
    assert extract_domain("https://www.reddit.com/r/python/") == "reddit.com"
    assert extract_domain("https://docs.python.org/3/") == "docs.python.org"


# ── Aggregator ────────────────────────────────────────────────────────────────

@pytest.fixture
def agg():
    return Aggregator()


def _make_raw(url: str, title: str = "T", snippet: str = "S", query: str = "q1") -> RawResult:
    return RawResult(title=title, snippet=snippet, url=url, query_source=query)


def test_dedup_exact_url(agg):
    results = [
        _make_raw("https://example.com/page", snippet="First snippet"),
        _make_raw("https://example.com/page", snippet="Second snippet"),
    ]
    out = agg.aggregate(results)
    assert len(out) == 1


def test_dedup_www_vs_no_www(agg):
    results = [
        _make_raw("https://www.example.com/page"),
        _make_raw("https://example.com/page"),
    ]
    out = agg.aggregate(results)
    assert len(out) == 1


def test_dedup_utm_params(agg):
    results = [
        _make_raw("https://example.com/page?utm_source=google"),
        _make_raw("https://example.com/page?utm_source=twitter"),
        _make_raw("https://example.com/page"),
    ]
    out = agg.aggregate(results)
    assert len(out) == 1


def test_preserves_unique_urls(agg):
    results = [_make_raw(f"https://example{i}.com/") for i in range(5)]
    out = agg.aggregate(results)
    assert len(out) == 5


def test_merges_query_sources(agg):
    results = [
        _make_raw("https://example.com/", query="query 1"),
        _make_raw("https://example.com/", query="query 2"),
    ]
    out = agg.aggregate(results)
    assert len(out) == 1
    assert "query 1" in out[0].query_sources
    assert "query 2" in out[0].query_sources


def test_roughly_half_dedup():
    """80 results with 40% duplication → ~50 unique."""
    agg = Aggregator()
    base_urls = [f"https://site{i}.com/page" for i in range(50)]
    dup_urls = [f"https://site{i}.com/page?utm_source=x" for i in range(30)]
    raw = [_make_raw(u) for u in base_urls + dup_urls]
    out = agg.aggregate(raw)
    assert len(out) == 50
