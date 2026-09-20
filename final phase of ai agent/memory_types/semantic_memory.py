"""
memory_types/semantic_memory.py
────────────────────────────────
Handles the data representation and metadata enrichment for
SEMANTIC memory entries.

Semantic memory stores facts, knowledge, concepts, definitions,
and general information — things that are context-independent and
universally applicable.

This module does NOT perform classification (that is done by
memory_type_analyzer.py using the Ollama LLM).  Its role is to
prepare a well-structured record ready for MongoDB insertion.
"""

from datetime import datetime, timezone
from typing import Any


MEMORY_TYPE = "SEMANTIC"

# Semantic-specific metadata descriptions
_SEMANTIC_DESCRIPTION = (
    "General factual knowledge, concept, definition, or universal truth "
    "that is context-independent and broadly applicable."
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
    Build a MongoDB-ready document for a SEMANTIC memory.

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
        "type_description": _SEMANTIC_DESCRIPTION,
    }
    if extra:
        record.update(extra)
    return record


def describe() -> str:
    """Return a human-readable description of this memory type."""
    return (
        f"[{MEMORY_TYPE}] {_SEMANTIC_DESCRIPTION}"
    )
