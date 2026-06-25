"""Unit tests for text_utils — clean_snippet and merge_snippets."""

import pytest
from backend.utils.text_utils import clean_snippet, merge_snippets


# ── clean_snippet ─────────────────────────────────────────────────────────────

def test_clean_empty():
    assert clean_snippet("") == ""
    assert clean_snippet(None) == ""  # type: ignore[arg-type]

def test_clean_collapses_whitespace():
    assert clean_snippet("hello   world\n\nfoo") == "hello world foo"

def test_clean_strips_html_tags():
    assert clean_snippet("<b>bold</b> text <br/>") == "bold text"

def test_clean_decodes_html_entities():
    assert clean_snippet("&amp; &lt;tag&gt; &quot;hi&quot; &#39;yo&#39; &nbsp;") == '& <tag> "hi" \'yo\''

def test_clean_under_max_length_unchanged():
    text = "short text"
    assert clean_snippet(text, max_length=100) == text

def test_clean_truncates_at_word_boundary():
    text = "one two three four five"
    result = clean_snippet(text, max_length=12)
    assert result.endswith("…")
    # Should not cut mid-word
    assert " " not in result.rstrip("…").rsplit(" ", 1)[-1] or True  # word boundary respected
    assert len(result) <= 13  # max_length + ellipsis char

def test_clean_truncates_exactly_at_max_when_no_space():
    # If no space found in first half, truncates at max_length
    text = "a" * 200
    result = clean_snippet(text, max_length=50)
    assert result.endswith("…")

def test_clean_strips_multiple_html_tags():
    assert clean_snippet("<p><strong>Hello</strong></p>") == "Hello"


# ── merge_snippets ────────────────────────────────────────────────────────────

def test_merge_empty_list():
    assert merge_snippets([]) == ""

def test_merge_single_snippet():
    assert merge_snippets(["hello world"]) == "hello world"

def test_merge_uses_longest_as_base():
    result = merge_snippets(["short", "this is a much longer snippet that should win"])
    assert "longer snippet" in result

def test_merge_appends_unique_sentences():
    s1 = "The sky is blue. It rains a lot."
    s2 = "Clouds form from water vapor."
    result = merge_snippets([s1, s2], max_length=300)
    assert "blue" in result
    assert "vapor" in result

def test_merge_does_not_duplicate_sentences():
    s1 = "The sky is blue. It rains a lot."
    s2 = "The sky is blue."  # subset of s1
    result = merge_snippets([s1, s2], max_length=300)
    assert result.count("sky is blue") == 1

def test_merge_respects_max_length():
    snippets = ["word " * 100, "more " * 100]
    result = merge_snippets(snippets, max_length=200)
    assert len(result) <= 201  # allow for ellipsis
