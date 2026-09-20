"""
memory_types/episodic_memory.py
────────────────────────────────
Handles the data representation and metadata enrichment for
EPISODIC memory entries.

Episodic memory stores specific events, experiences, decisions,
conversations, and interactions that are tied to a specific time,
place, or context.

This module does NOT perform classification (that is done by
memory_type_analyzer.py using the Ollama LLM).  Its role is to
prepare a well-structured record ready for MongoDB insertion.
"""

from datetime import datetime, timezone
from typing import Any


MEMORY_TYPE = "EPISODIC"

_EPISODIC_DESCRIPTION = (
    "A specific event, experience, interaction, decision, or conversation "
    "tied to a particular time, place, or context."
)


def build_record(
    memory: str,
    context_score: float,
    importance_score: float,
    persistence_score: float,
    confidence_score: float,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict:
    """
    Build a MongoDB-ready document for an EPISODIC memory.

    Parameters
    ----------
    memory            : str   — the original user statement
    context_score     : float — from context_analysis
    importance_score  : float — from importance_analysis
    persistence_score : float — from persistence_analysis
    confidence_score  : float — from Ollama classification
    reason            : str   — LLM classification explanation
    extra             : dict  — optional additional fields

    Returns
    -------
    dict — structured memory document (not yet inserted into MongoDB)
    """
    record = {
        "memory": memory.strip(),
        "memory_type": MEMORY_TYPE,
        "context_score": round(float(context_score), 4),
        "importance_score": round(float(importance_score), 4),
        "persistence_score": round(float(persistence_score), 4),
        "confidence_score": round(float(confidence_score), 4),
        "reason": reason.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type_description": _EPISODIC_DESCRIPTION,
    }
    if extra:
        record.update(extra)
    return record


def describe() -> str:
    """Return a human-readable description of this memory type."""
    return f"[{MEMORY_TYPE}] {_EPISODIC_DESCRIPTION}"
