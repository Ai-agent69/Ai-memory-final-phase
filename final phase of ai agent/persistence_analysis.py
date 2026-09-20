"""
persistence_analysis.py
────────────────────────
Analyzes whether a piece of information is likely to remain useful and
accurate over a long period of time.

Produces a persistence_score in the range [0.0, 1.0].

High-scoring inputs are timeless facts, stable procedures, or enduring
knowledge.  Low-scoring inputs are highly time-sensitive, conversational,
or temporary.

Score interpretation:
    0.0 – 0.39  : Ephemeral / short-lived — not worth long-term storage
    0.40 – 0.69 : Moderately persistent
    0.70 – 1.0  : Highly persistent — very good long-term memory candidate
"""

import os
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PERSISTENCE_THRESHOLD: float = float(os.getenv("PERSISTENCE_THRESHOLD", "0.4"))

# ── Signals that suggest HIGH persistence ──────────────────────────────────
_HIGH_PERSISTENCE_SIGNALS: list[str] = [
    # definitional / factual
    "is a", "is an", "is the", "are a", "defined as", "means",
    "refers to", "known as", "stands for", "called", "oriented",
    "described as",
    # universal truth / stable facts
    "always", "never", "every time", "in general", "generally",
    "by definition", "fundamentally", "principle", "law", "rule",
    "theory", "concept", "fact",
    # stable skills / procedures
    "how to", "procedure", "method", "approach", "technique",
    "workflow", "process", "pattern", "best practice", "standard",
    "algorithm", "formula", "step", "steps",
    "first", "then", "next", "finally",
    # procedural action verbs (durable instructions)
    "install", "create", "connect", "configure", "setup", "set up",
    "implement", "build", "deploy", "initialize", "initialise",
    "run", "execute", "define", "declare", "import", "enable",
    "analyze", "calculate", "classify", "store", "memory",
    # enduring knowledge domains
    "architecture", "design", "protocol", "specification", "schema",
    "api", "interface", "framework", "library", "language", "syntax",
    "database", "model", "system", "structure",
    "nosql", "sql", "document", "collection", "query", "storage",
    "programming", "function", "class", "module", "server", "client",
]

# ── Signals that suggest LOW persistence (ephemeral content) ───────────────
_LOW_PERSISTENCE_SIGNALS: list[str] = [
    "today", "tomorrow", "yesterday", "right now", "currently",
    "at the moment", "this week", "next week", "last week",
    "this month", "next month", "soon", "later today",
    "just happened", "just now", "breaking", "latest update",
    "temporary", "temporarily", "for now", "short-term",
    "deadline", "due date", "meeting at",
]

# ── Time-bound patterns ────────────────────────────────────────────────────
_TIME_BOUND_RE = re.compile(
    r"\b(\d{1,2}[:/]\d{2}|"          # times like 3:00 or 15:30
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"  # dates like 12/05/2024
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)

# ── Opinion/sentiment markers (low persistence) ────────────────────────────
_OPINION_RE = re.compile(
    r"\b(i feel|i think|i believe|in my opinion|personally|"
    r"i prefer|i like|i hate|i love|i want)\b",
    re.IGNORECASE,
)


def analyze_persistence(user_input: str) -> dict:
    """
    Assess whether *user_input* represents durable, long-term-useful knowledge.

    Parameters
    ----------
    user_input : str
        The raw text from the user.

    Returns
    -------
    dict with keys:
        persistence_score : float  — 0.0 to 1.0
        is_persistent     : bool   — True if score >= PERSISTENCE_THRESHOLD
        reason            : str    — human-readable explanation
    """
    text = (user_input or "").strip()

    # ── Guard ─────────────────────────────────────────────────────────────
    if not text:
        return {
            "persistence_score": 0.0,
            "is_persistent": False,
            "reason": "Empty input — nothing to analyze.",
        }

    lower = text.lower()
    score = 0.0
    reasons: list[str] = []

    # ── High-persistence signals ───────────────────────────────────────────
    found_high = [s for s in _HIGH_PERSISTENCE_SIGNALS if s in lower]
    if found_high:
        bonus = min(0.50, len(found_high) * 0.10)
        score += bonus
        reasons.append(f"Persistent-knowledge signals ({len(found_high)}): {', '.join(found_high[:3])} → +{bonus:.2f}.")

    # ── Low-persistence / time-bound signals ──────────────────────────────
    found_low = [s for s in _LOW_PERSISTENCE_SIGNALS if s in lower]
    if found_low:
        penalty = min(0.40, len(found_low) * 0.10)
        score -= penalty
        reasons.append(f"Ephemeral/time-bound signals ({found_low[:3]}) → -{penalty:.2f}.")

    # ── Time-bound date/time patterns ─────────────────────────────────────
    time_matches = _TIME_BOUND_RE.findall(text)
    if time_matches:
        penalty = min(0.20, len(time_matches) * 0.07)
        score -= penalty
        reasons.append(f"Specific date/time reference detected → -{penalty:.2f}.")

    # ── Personal opinions (low persistence) ───────────────────────────────
    opinion_matches = _OPINION_RE.findall(text)
    if opinion_matches:
        score -= 0.15
        reasons.append("Personal opinion/preference detected (-0.15).")

    # ── Length bonus (longer statements tend to be more substantive) ───────
    words = len(text.split())
    length_bonus = min(0.15, words * 0.01)
    score += length_bonus
    reasons.append(f"Length ({words} words) → +{length_bonus:.2f}.")

    # ── Clamp ─────────────────────────────────────────────────────────────
    score = round(min(1.0, max(0.0, score)), 4)
    is_persistent = score >= PERSISTENCE_THRESHOLD

    return {
        "persistence_score": score,
        "is_persistent": is_persistent,
        "reason": " ".join(reasons) if reasons else "No strong persistence signals found.",
    }
