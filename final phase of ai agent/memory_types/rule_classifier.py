"""
memory_types/rule_classifier.py
───────────────────────────────
Deterministic keyword and pattern-based classifier for memory types.

Classifies user statements into:
    - SEMANTIC   : Facts, identity, preferences, skills, education, persistent projects.
    - EPISODIC   : Events, past experiences, time-anchored occurrences.
    - PROCEDURAL : Methods, instructions, workflows, step-by-step how-to.

Produces:
    memory_type : str | None
    confidence  : float (0.0 to 1.0)
    is_clear    : bool  (True if confidence >= RULE_CONFIDENCE_THRESHOLD)
    matched_rule: str   (description of the rule that triggered)

If is_clear is False, the pipeline treats the input as UNCLEAR and invokes
Ollama as an ambiguous-case fallback.
"""

import os
import re
from typing import Optional

# Configurable confidence threshold for a "CLEAR" determination
RULE_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("RULE_CONFIDENCE_THRESHOLD", "0.70")
)

# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

# SEMANTIC patterns (facts, identity, knowledge, skills, education, projects)
_SEMANTIC_PATTERNS = [
    # Identity & Names
    (re.compile(r"\b(my name is|my name's|call me|i am called|i'm called)\b", re.I), 0.95, "Identity / name statement"),
    (re.compile(r"\bi am ([A-Z][a-z]+(\s+[A-Z][a-z]+)+)\b"), 0.90, "Personal full name declaration"),
    
    # Occupation, Role & Education
    (re.compile(r"\b(i am a|i am an|i'm a|i'm an)\s+(b\.?tech|student|engineer|developer|programmer|researcher|teacher|professor|scientist|designer|doctor|analyst)\b", re.I), 0.95, "Occupation or student role"),
    (re.compile(r"\b(i study|i am studying|i'm studying|my college|my university|my school|my degree|my major|my branch)\b", re.I), 0.92, "Educational background"),
    (re.compile(r"\bi work as\b", re.I), 0.90, "Professional role declaration"),
    
    # Skills, Languages & Technologies
    (re.compile(r"\b(i use|i know|i code in|i program in)\s+(python|java|c\+\+|javascript|typescript|go|rust|mongodb|sql|react|docker|git|linux)\b", re.I), 0.92, "Technical skill or tool usage"),
    (re.compile(r"\b(my skill|my skills|proficient in|experienced in|experience with)\b", re.I), 0.90, "Skill declaration"),
    
    # Persistent Projects & Preparations
    (re.compile(r"\b(i am working on|i'm working on|i am developing|i'm building|my project is)\b", re.I), 0.90, "Persistent project commitment"),
    (re.compile(r"\b(i am preparing for|i'm preparing for|preparing for)\s+(gate|gre|cat|upsc|exams?|interviews?)\b", re.I), 0.95, "Long-term goal / exam preparation"),
    
    # Preferences & Opinions
    (re.compile(r"\b(my preference|i prefer|i like|i love|my favorite|my favourite)\b", re.I), 0.88, "User preference"),
    
    # Factual Definitions & Concepts
    (re.compile(r"\b(is a|is an|are a|are an|defined as|refers to|means that|known as|stands for)\b", re.I), 0.88, "Factual definition or concept"),
    (re.compile(r"\b(is the|are the|always|never|every)\b", re.I), 0.75, "Universal or general fact"),
]

# EPISODIC patterns (events, experiences, specific past/present moments)
_EPISODIC_PATTERNS = [
    # Temporal event markers
    (re.compile(r"\b(yesterday|last week|last month|last year|last night|a few days ago|recently|on monday|on tuesday|on wednesday|on thursday|on friday|on saturday|on sunday)\b", re.I), 0.92, "Past temporal anchor"),
    (re.compile(r"\b(today|this morning|this afternoon|earlier today)\b", re.I), 0.82, "Current day event anchor"),
    
    # Event experiences & occurrences
    (re.compile(r"\b(i went|i visited|i attended|i experienced|i participated in|i joined)\b", re.I), 0.92, "Past personal event experience"),
    (re.compile(r"\b(i met|we decided|we discussed|we met|team decided)\b", re.I), 0.90, "Interactive episode or team decision"),
    (re.compile(r"\b(happened|occurred|took place)\b", re.I), 0.85, "Event occurrence"),
    (re.compile(r"\b(attended a|went to a)\s+(hackathon|conference|meetup|meeting|workshop|summit|concert|party|webinar)\b", re.I), 0.95, "Specific event attendance"),
]

# PROCEDURAL patterns (methods, instructions, workflows, step-by-step how-to)
_PROCEDURAL_PATTERNS = [
    # How-to & guidance
    (re.compile(r"\b(how to|how do i|how can i|instructions? for|guide to)\b", re.I), 0.95, "How-to instruction phrase"),
    
    # Step-by-step sequences
    (re.compile(r"\b(first\b.*?\bthen\b|first\b.*?\bnext\b|step \d|step-by-step)\b", re.I), 0.95, "Sequential step sequence"),
    (re.compile(r"^(first|step 1|1\.)\s+", re.I), 0.92, "Opening sequential step"),
    (re.compile(r"\b(then|next|finally|after that)\b", re.I), 0.78, "Sequential step connector"),
    
    # Process & methodology terms
    (re.compile(r"\b(procedure|method|process|workflow|algorithm|recipe|instructions?|steps? to)\b", re.I), 0.85, "Methodological terminology"),
    
    # Imperative command sequences (e.g., "activate ..., then run ...")
    (re.compile(r"\b(activate|install|run|deploy|configure|execute|initialize|build)\b.*?\b(then|next|after|and then)\b", re.I), 0.92, "Command execution sequence"),
]


# ---------------------------------------------------------------------------
# Classification Function
# ---------------------------------------------------------------------------

def classify_by_rules(text: str) -> dict:
    """
    Classify the memory type of text using deterministic keyword/pattern rules.

    Parameters
    ----------
    text : str
        The input statement.

    Returns
    -------
    dict with keys:
        memory_type  : str | None ("SEMANTIC", "EPISODIC", "PROCEDURAL", or None)
        confidence   : float (0.0 to 1.0)
        is_clear     : bool (True if confidence >= RULE_CONFIDENCE_THRESHOLD)
        matched_rule : str
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "memory_type": None,
            "confidence": 0.0,
            "is_clear": False,
            "matched_rule": "Empty input",
        }

    # Evaluate matches across all categories
    candidates: list[tuple[str, float, str]] = []

    # 1. Check PROCEDURAL
    for pattern, weight, rule_desc in _PROCEDURAL_PATTERNS:
        if pattern.search(cleaned):
            candidates.append(("PROCEDURAL", weight, rule_desc))
            break

    # 2. Check EPISODIC
    for pattern, weight, rule_desc in _EPISODIC_PATTERNS:
        if pattern.search(cleaned):
            candidates.append(("EPISODIC", weight, rule_desc))
            break

    # 3. Check SEMANTIC
    for pattern, weight, rule_desc in _SEMANTIC_PATTERNS:
        if pattern.search(cleaned):
            candidates.append(("SEMANTIC", weight, rule_desc))
            break

    if not candidates:
        # No pattern matched -> UNCLEAR
        return {
            "memory_type": None,
            "confidence": 0.0,
            "is_clear": False,
            "matched_rule": "No deterministic rule matched",
        }

    # If only one category matched, that is our candidate
    if len(candidates) == 1:
        mem_type, confidence, rule_desc = candidates[0]
        return {
            "memory_type": mem_type,
            "confidence": confidence,
            "is_clear": confidence >= RULE_CONFIDENCE_THRESHOLD,
            "matched_rule": rule_desc,
        }

    # If multiple categories matched, determine precedence or conflict
    # E.g., "Yesterday I attended a hackathon and learned Python" has both EPISODIC and SEMANTIC.
    # An event with past temporal anchor ('yesterday') is primarily EPISODIC.
    candidates.sort(key=lambda c: c[1], reverse=True)
    best_type, best_conf, best_rule = candidates[0]
    second_type, second_conf, _ = candidates[1]

    # If top two candidates are very close in confidence, it's ambiguous -> UNCLEAR
    if abs(best_conf - second_conf) < 0.05 and best_type != second_type:
        return {
            "memory_type": best_type,
            "confidence": 0.50,
            "is_clear": False,
            "matched_rule": f"Ambiguous between {best_type} and {second_type}",
        }

    return {
        "memory_type": best_type,
        "confidence": best_conf,
        "is_clear": best_conf >= RULE_CONFIDENCE_THRESHOLD,
        "matched_rule": best_rule,
    }
