"""
URL normalization utilities.

All functions are pure (no side effects) and safe to call in a hot path.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Tracking / UTM parameters to strip
_STRIP_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "ref",
        "source",
        "referrer",
        "fbclid",
        "gclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "igshid",
        "twclid",
    }
)


def normalize_url(url: str) -> str:
    """
    Return a canonical form of *url* for deduplication purposes.

    Transformations applied:
    - Lowercase the scheme and host
    - Strip ``www.`` prefix from the host
    - Remove tracking / UTM query parameters
    - Strip trailing slash from the path (unless path is just ``/``)
    - Sort remaining query parameters for stability
    """
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.lower().strip()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip www. prefix
    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Strip tracking params, sort the rest
    params = parse_qs(parsed.query, keep_blank_values=False)
    cleaned = {k: v for k, v in params.items() if k not in _STRIP_PARAMS}
    query = urlencode(sorted(cleaned.items()), doseq=True)

    # Strip trailing slash (but keep root "/" intact)
    path = parsed.path.rstrip("/") or "/"

    canonical = urlunparse((scheme, netloc, path, parsed.params, query, ""))
    return canonical


def extract_domain(url: str) -> str:
    """Return the bare domain (no www, no scheme) from a URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # Strip port if present
        return netloc.split(":")[0]
    except Exception:
        return url
