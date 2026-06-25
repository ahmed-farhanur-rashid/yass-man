"""
Text cleaning and manipulation utilities.
"""

from __future__ import annotations

import re


def clean_snippet(text: str, max_length: int = 500) -> str:
    """
    Clean and truncate a search result snippet.

    - Collapses whitespace and newlines
    - Strips HTML entities and common residue
    - Truncates to *max_length* characters at a word boundary
    """
    if not text:
        return ""

    # Strip basic HTML tags that sneak through
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )

    # Collapse whitespace after tag removal and entity decoding
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_length:
        return text

    # Truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


def merge_snippets(snippets: list[str], max_length: int = 600) -> str:
    """
    Merge multiple snippets for the same URL into one.

    Strategy:
    1. Start with the longest snippet.
    2. Append unique sentences from shorter ones until the budget is full.
    """
    if not snippets:
        return ""
    if len(snippets) == 1:
        return clean_snippet(snippets[0], max_length)

    # Use the longest as the base
    base = max(snippets, key=len)
    base_clean = clean_snippet(base, max_length)

    seen_sentences: set[str] = set()
    for sent in _split_sentences(base_clean):
        seen_sentences.add(sent.strip().lower())

    extras: list[str] = []
    for snippet in snippets:
        if snippet == base:
            continue
        for sent in _split_sentences(snippet):
            norm = sent.strip().lower()
            if norm and norm not in seen_sentences:
                seen_sentences.add(norm)
                extras.append(sent.strip())

    merged = base_clean
    for extra in extras:
        candidate = merged + " " + extra
        if len(candidate) <= max_length:
            merged = candidate
        else:
            break

    return merged


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter on period / exclamation / question mark."""
    return re.split(r"(?<=[.!?])\s+", text)

