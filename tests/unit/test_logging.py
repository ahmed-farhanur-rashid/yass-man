"""Unit tests for QueryLogger and FeedbackLogger."""

import json
import time
from pathlib import Path

import pytest

from backend.logging.feedback import FeedbackLogger
from backend.logging.query_logger import QueryLogger


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


# ── QueryLogger ───────────────────────────────────────────────────────────────

def test_query_logger_creates_directory(log_dir):
    QueryLogger(log_dir)
    assert log_dir.exists()


def test_query_logger_writes_jsonl_record(log_dir):
    ql = QueryLogger(log_dir)
    ql.log(
        query_id="test-uuid",
        timestamp="2026-01-01T00:00:00+00:00",
        query="what is entropy",
        expanded_queries=["what is entropy", "entropy definition"],
        num_results_retrieved=20,
        num_results_after_dedup=15,
        num_results_after_clustering=10,
        top_k_urls=["https://a.com/", "https://b.com/"],
        reranker_scores=[0.95, 0.88],
        answer_generated=True,
        latency_ms={"router": 2.0, "total": 850.0},
    )
    files = list(log_dir.glob("queries-*.jsonl"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(records) == 1
    r = records[0]
    assert r["query_id"] == "test-uuid"
    assert r["query"] == "what is entropy"
    assert r["num_results_retrieved"] == 20
    assert r["answer_generated"] is True


def test_query_logger_multiple_records_same_file(log_dir):
    ql = QueryLogger(log_dir)
    for i in range(3):
        ql.log(
            query_id=f"uuid-{i}",
            timestamp="2026-01-01T00:00:00+00:00",
            query=f"query {i}",
            expanded_queries=[f"query {i}"],
            num_results_retrieved=10,
            num_results_after_dedup=8,
            num_results_after_clustering=6,
            top_k_urls=[],
            reranker_scores=[],
            answer_generated=False,
            latency_ms={"total": 100.0},
        )
    files = list(log_dir.glob("queries-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 3


def test_query_logger_records_are_valid_json(log_dir):
    ql = QueryLogger(log_dir)
    ql.log(
        query_id="x",
        timestamp="2026-01-01T00:00:00+00:00",
        query="unicode query: 日本語",
        expanded_queries=[],
        num_results_retrieved=0,
        num_results_after_dedup=0,
        num_results_after_clustering=0,
        top_k_urls=[],
        reranker_scores=[],
        answer_generated=False,
        latency_ms={},
    )
    files = list(log_dir.glob("queries-*.jsonl"))
    record = json.loads(files[0].read_text().strip())
    assert record["query"] == "unicode query: 日本語"


# ── FeedbackLogger ────────────────────────────────────────────────────────────

def test_feedback_logger_creates_directory(log_dir):
    FeedbackLogger(log_dir)
    assert log_dir.exists()


def test_feedback_logger_writes_thumbs_up(log_dir):
    fl = FeedbackLogger(log_dir)
    fl.log_feedback(query_id="qid-1", result_url="https://example.com/", signal="up")
    records = [json.loads(l) for l in (log_dir / "feedback.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["signal"] == "up"
    assert records[0]["query_id"] == "qid-1"
    assert "timestamp" in records[0]


def test_feedback_logger_writes_thumbs_down(log_dir):
    fl = FeedbackLogger(log_dir)
    fl.log_feedback(query_id="qid-2", result_url="https://example.com/", signal="down")
    record = json.loads((log_dir / "feedback.jsonl").read_text().strip())
    assert record["signal"] == "down"


def test_click_logger_writes_to_clicks_file(log_dir):
    fl = FeedbackLogger(log_dir)
    fl.log_click(query_id="qid-3", result_url="https://clicked.com/page")
    record = json.loads((log_dir / "clicks.jsonl").read_text().strip())
    assert record["query_id"] == "qid-3"
    assert record["result_url"] == "https://clicked.com/page"
    assert "timestamp" in record


def test_feedback_and_clicks_in_separate_files(log_dir):
    fl = FeedbackLogger(log_dir)
    fl.log_feedback(query_id="q1", result_url="https://a.com/", signal="up")
    fl.log_click(query_id="q2", result_url="https://b.com/")
    assert (log_dir / "feedback.jsonl").exists()
    assert (log_dir / "clicks.jsonl").exists()


def test_multiple_clicks_append(log_dir):
    fl = FeedbackLogger(log_dir)
    for i in range(4):
        fl.log_click(query_id=f"q{i}", result_url=f"https://site{i}.com/")
    lines = (log_dir / "clicks.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
