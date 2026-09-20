"""
longterm_memory.py
───────────────────
Combines context, importance, and persistence scores to make the binary
long-term memory decision:

    Should this information be stored as long-term memory?

This is the gate that determines whether the expensive Ollama LLM call
should be made.  If the decision is False, the information is not stored
and no LLM call occurs.

Decision logic:
    A weighted average of the three scores is computed.  If the weighted
    average exceeds the LONGTERM_THRESHOLD, the decision is True.

    Weights (defaults):
        context     : 0.30   — relevance to current conversation
        importance  : 0.40   — value of the information itself
        persistence : 0.30   — durability / long-term usefulness

    At least one individual score must also exceed its own threshold
    to prevent cases where three mediocre scores average above threshold
    while all individual analyzers would reject the input.
"""

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LONGTERM_THRESHOLD: float = float(os.getenv("LONGTERM_THRESHOLD", "0.5"))

# Weights for the weighted average
_WEIGHT_CONTEXT: float = 0.30
_WEIGHT_IMPORTANCE: float = 0.40
_WEIGHT_PERSISTENCE: float = 0.30


def decide_longterm(
    context_score: float,
    importance_score: float,
    persistence_score: float,
) -> dict:
    """
    Decide whether information should be stored as long-term memory.

    Parameters
    ----------
    context_score     : float — output of context_analysis.analyze_context()
    importance_score  : float — output of importance_analysis.analyze_importance()
    persistence_score : float — output of persistence_analysis.analyze_persistence()

    Returns
    -------
    dict with keys:
        is_longterm     : bool  — True → proceed to LLM classification
        combined_score  : float — weighted average of the three scores
        reason          : str   — human-readable explanation
    """
    # Clamp inputs to [0, 1]
    ctx = max(0.0, min(1.0, context_score))
    imp = max(0.0, min(1.0, importance_score))
    per = max(0.0, min(1.0, persistence_score))

    combined = round(
        ctx * _WEIGHT_CONTEXT
        + imp * _WEIGHT_IMPORTANCE
        + per * _WEIGHT_PERSISTENCE,
        4,
    )

    # At least one score must individually exceed threshold to avoid
    # three weak scores collectively passing the gate.
    any_strong = (ctx >= LONGTERM_THRESHOLD
                  or imp >= LONGTERM_THRESHOLD
                  or per >= LONGTERM_THRESHOLD)

    is_longterm = (combined >= LONGTERM_THRESHOLD) and any_strong

    reason_parts: list[str] = [
        f"Weighted score: {combined:.4f} "
        f"(context={ctx:.2f}×{_WEIGHT_CONTEXT}, "
        f"importance={imp:.2f}×{_WEIGHT_IMPORTANCE}, "
        f"persistence={per:.2f}×{_WEIGHT_PERSISTENCE}).",
    ]
    if not (combined >= LONGTERM_THRESHOLD):
        reason_parts.append(
            f"Combined score {combined:.4f} < threshold {LONGTERM_THRESHOLD}."
        )
    if not any_strong:
        reason_parts.append(
            "No individual score exceeds the long-term threshold."
        )
    if is_longterm:
        reason_parts.append("Decision: LONG-TERM — will classify memory type.")
    else:
        reason_parts.append("Decision: NOT long-term — will not be stored.")

    return {
        "is_longterm": is_longterm,
        "combined_score": combined,
        "reason": " ".join(reason_parts),
    }
