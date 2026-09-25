"""
memory_evaluator.py
-------------------
Python-owned routing, scoring, and storage decisions for analyzed memories.

Ollama supplies the memory type, memory scope, and analytical scores only for
potential long-term memories. This module skips scoring for temporary inputs,
calculates the final score for candidates, and makes the deterministic
threshold decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LONG_TERM_THRESHOLD: float = 0.70

IMPORTANCE_WEIGHT = 0.35
PERSISTENCE_WEIGHT = 0.35
USEFULNESS_WEIGHT = 0.30
TEMPORARY_SCOPE = "TEMPORARY"
POTENTIAL_LONG_TERM_SCOPE = "POTENTIAL_LONG_TERM"


def _validate_score(score: Any, field_name: str) -> float:
    if isinstance(score, bool):
        raise ValueError(f"{field_name} must be numeric, got boolean.")
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0.")
    return value


def _validate_threshold(threshold: Any) -> float:
    return _validate_score(threshold, "threshold")


def calculate_final_score(
    importance_score: float,
    persistence_score: float,
    usefulness_score: float,
) -> float:
    """Calculate the final memory score using the required weighted formula."""
    importance = _validate_score(importance_score, "importance_score")
    persistence = _validate_score(persistence_score, "persistence_score")
    usefulness = _validate_score(usefulness_score, "usefulness_score")
    final_score = (
        importance * IMPORTANCE_WEIGHT
        + persistence * PERSISTENCE_WEIGHT
        + usefulness * USEFULNESS_WEIGHT
    )
    return round(final_score, 4)


def make_memory_decision(
    analysis: dict[str, Any],
    threshold: float = LONG_TERM_THRESHOLD,
) -> dict[str, Any]:
    """
    Decide whether a validated analysis should be stored long-term.

    TEMPORARY scope goes directly to Trash. Only POTENTIAL_LONG_TERM scope is
    scored. The Ollama long_term_beneficial field is retained as an analytical
    signal, but Python's final score and threshold are authoritative.
    """
    memory_scope = (analysis.get("memory_scope") or "").strip().upper()
    normalized_threshold = _validate_threshold(threshold)

    if memory_scope == TEMPORARY_SCOPE:
        return {
            "final_score": None,
            "threshold": normalized_threshold,
            "decision": "TRASH",
            "is_longterm": False,
            "scoring_skipped": True,
            "reason": "Temporary memory scope; scoring skipped.",
        }

    if memory_scope != POTENTIAL_LONG_TERM_SCOPE:
        raise ValueError(
            "memory_scope must be TEMPORARY or POTENTIAL_LONG_TERM before "
            "making a memory decision."
        )

    final_score = calculate_final_score(
        analysis["importance_score"],
        analysis["persistence_score"],
        analysis["usefulness_score"],
    )
    is_longterm = final_score >= normalized_threshold
    decision = "LONG_TERM" if is_longterm else "TRASH"

    return {
        "final_score": final_score,
        "threshold": normalized_threshold,
        "decision": decision,
        "is_longterm": is_longterm,
        "scoring_skipped": False,
        "reason": (
            f"Final score {final_score:.4f} "
            f"{'>=' if is_longterm else '<'} threshold {normalized_threshold:.2f}."
        ),
    }


def build_memory_document(
    memory_text: str,
    analysis: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Build the MongoDB document for a validated long-term memory."""
    text = (memory_text or "").strip()
    return {
        "memory_text": text,
        "memory": text,
        "memory_type": analysis["memory_type"],
        "memory_scope": analysis["memory_scope"],
        "long_term_beneficial": analysis["long_term_beneficial"],
        "importance_score": round(float(analysis["importance_score"]), 4),
        "persistence_score": round(float(analysis["persistence_score"]), 4),
        "usefulness_score": round(float(analysis["usefulness_score"]), 4),
        "final_score": round(float(decision["final_score"]), 4),
        "decision": decision["decision"],
        "reason": analysis["reason"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
