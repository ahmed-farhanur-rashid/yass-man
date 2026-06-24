"""
Query Logger — Phase 9.

Writes one JSONL record per completed search to {LOG_DIR}/queries-{date}.jsonl.
Rotates daily. Never deletes old logs.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryLogger:
    """
    Append-only JSONL logger for search queries.

    Thread-safe for asyncio (file I/O is synchronous but fast).
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        today = date.today().isoformat()
        return self._log_dir / f"queries-{today}.jsonl"

    def log(
        self,
        *,
        query_id: str,
        timestamp: str,
        query: str,
        expanded_queries: list[str],
        num_results_retrieved: int,
        num_results_after_dedup: int,
        num_results_after_clustering: int,
        top_k_urls: list[str],
        reranker_scores: list[float],
        answer_generated: bool,
        latency_ms: dict[str, float],
    ) -> None:
        """Append one query record to today's log file."""
        record = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query": query,
            "expanded_queries": expanded_queries,
            "num_results_retrieved": num_results_retrieved,
            "num_results_after_dedup": num_results_after_dedup,
            "num_results_after_clustering": num_results_after_clustering,
            "top_k_urls": top_k_urls,
            "reranker_scores": reranker_scores,
            "answer_generated": answer_generated,
            "latency_ms": latency_ms,
        }
        try:
            with self._log_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("QueryLogger: failed to write log record: %s", exc)
