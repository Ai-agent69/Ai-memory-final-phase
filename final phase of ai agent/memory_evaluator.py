"""
memory_evaluator.py
-------------------
Python-owned scoring and storage decisions for analyzed memories.

Ollama supplies validated analytical scores. This module calculates the final
score and makes the deterministic threshold decision.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

LONG_TERM_THRESHOLD: float = float(os.getenv("LONG_TERM_THRESHOLD", "0.70"))

IMPORTANCE_WEIGHT = 0.35
PERSISTENCE_WEIGHT = 0.35
USEFULNESS_WEIGHT = 0.30


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def calculate_final_score(
    importance_score: float,
    persistence_score: float,
    usefulness_score: float,
) -> float:
    """Calculate the final memory score using the required weighted formula."""
    final_score = (
        _clamp_score(importance_score) * IMPORTANCE_WEIGHT
        + _clamp_score(persistence_score) * PERSISTENCE_WEIGHT
        + _clamp_score(usefulness_score) * USEFULNESS_WEIGHT
    )
    return round(_clamp_score(final_score), 4)


def make_memory_decision(
    analysis: dict[str, Any],
    threshold: float = LONG_TERM_THRESHOLD,
) -> dict[str, Any]:
    """
    Decide whether a validated analysis should be stored long-term.

    The Ollama long_term_beneficial field is retained as an analytical signal,
    but the final score and threshold are authoritative.
    """
    final_score = calculate_final_score(
        analysis["importance_score"],
        analysis["persistence_score"],
        analysis["usefulness_score"],
    )
    normalized_threshold = _clamp_score(threshold)
    is_longterm = final_score >= normalized_threshold
    decision = "LONG_TERM" if is_longterm else "TEMPORARY"

    return {
        "final_score": final_score,
        "threshold": normalized_threshold,
        "decision": decision,
        "is_longterm": is_longterm,
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
        "long_term_beneficial": analysis["long_term_beneficial"],
        "importance_score": round(float(analysis["importance_score"]), 4),
        "persistence_score": round(float(analysis["persistence_score"]), 4),
        "usefulness_score": round(float(analysis["usefulness_score"]), 4),
        "final_score": round(float(decision["final_score"]), 4),
        "decision": decision["decision"],
        "reason": analysis["reason"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
