"""
Feedback & Click Logger — Phase 9.

POST /feedback → {LOG_DIR}/feedback.jsonl
POST /click    → {LOG_DIR}/clicks.jsonl
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class FeedbackLogger:
    """Append-only loggers for feedback signals and click-throughs."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._feedback_path = log_dir / "feedback.jsonl"
        self._clicks_path = log_dir / "clicks.jsonl"

    def log_feedback(self, *, query_id: str, result_url: str, signal: str) -> None:
        """Append a thumbs-up/down signal."""
        record = {
            "query_id": query_id,
            "result_url": result_url,
            "signal": signal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append(self._feedback_path, record)

    def log_click(self, *, query_id: str, result_url: str) -> None:
        """Append a click-through event."""
        record = {
            "query_id": query_id,
            "result_url": result_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append(self._clicks_path, record)

    def _append(self, path: Path, record: dict) -> None:
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("FeedbackLogger: failed to write to %s: %s", path, exc)
