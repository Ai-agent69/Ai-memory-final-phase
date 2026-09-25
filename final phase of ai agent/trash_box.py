"""
trash_box.py
------------
Session-level Trash Box for memories that should not be stored in MongoDB.

Trash entries are intentionally in-memory only. They make the routing decision
explicit for the running application without adding another persistence layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class TrashBox:
    """In-memory record of discarded memory-analysis inputs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(
        self,
        *,
        input_text: str,
        memory_type: str,
        memory_scope: str,
        reason: str,
        final_score: float | None = None,
        threshold: float | None = None,
        scoring_skipped: bool = False,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "input": input_text,
            "memory_type": memory_type,
            "memory_scope": memory_scope,
            "reason": reason,
            "decision": "TRASH",
            "scoring_skipped": scoring_skipped,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if final_score is not None:
            entry["final_score"] = round(float(final_score), 4)
        if threshold is not None:
            entry["threshold"] = round(float(threshold), 4)

        self._entries.append(entry)
        return entry

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


TRASH_BOX = TrashBox()


def record_trash(**kwargs: Any) -> dict[str, Any]:
    """Record a discarded input in the process-level Trash Box."""
    return TRASH_BOX.record(**kwargs)
