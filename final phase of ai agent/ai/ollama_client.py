"""
ai/ollama_client.py
────────────────────
Clean, reusable client for communicating with the local Ollama service.

Responsibilities:
    - Read model configuration from environment (OLLAMA_MODEL).
    - Send classification prompts and receive structured JSON responses.
    - Validate the JSON response (memory_type, confidence, reason).
    - Handle all error conditions gracefully without crashing.

The Ollama Python package (v0.6.2) is used for communication.
All Ollama API calls are centralised here — no other module calls Ollama.

Environment variables:
    OLLAMA_MODEL      — model name (default: gemma4:12b)
    OLLAMA_BASE_URL   — Ollama host (default: http://localhost:11434)
"""

import json
import logging
import os
import re
from typing import Optional

import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

VALID_MEMORY_TYPES = {"SEMANTIC", "EPISODIC", "PROCEDURAL"}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_CLASSIFICATION_PROMPT = """\
You are a memory classification system. Analyze the following statement and \
classify it into exactly ONE of these memory types:

- SEMANTIC   : Facts, knowledge, concepts, definitions, general information.
               Example: "MongoDB is a document-oriented database."

- EPISODIC   : Specific events, experiences, interactions, decisions, \
conversations tied to a time or context.
               Example: "Yesterday our team decided to move to MongoDB."

- PROCEDURAL : Skills, procedures, methods, workflows — describes HOW to do \
something.
               Example: "First analyze the memory, then classify it, then \
store it in MongoDB."

{context_section}
Statement to classify:
"{statement}"

Respond ONLY with valid JSON in this exact format (no extra text, no markdown):
{{
    "memory_type": "SEMANTIC",
    "confidence": 0.92,
    "reason": "Brief explanation of why this memory type was chosen."
}}

Rules:
- memory_type must be exactly one of: SEMANTIC, EPISODIC, PROCEDURAL
- confidence must be a float between 0.0 and 1.0
- reason must be a non-empty string
- Do not include anything outside the JSON object
"""

_CONTEXT_SECTION_TEMPLATE = """\
Relevant previous memories for context:
{memories}

Use these memories only if they help understand the statement above.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompt(statement: str, relevant_memories: Optional[list[dict]] = None) -> str:
    """Build the classification prompt, optionally including retrieved memories."""
    if relevant_memories:
        memory_lines = "\n".join(
            f'- [{m.get("memory_type", "?")}] {m.get("memory", "")}' 
            for m in relevant_memories[:5]  # cap at 5 to keep prompt short
        )
        context_section = _CONTEXT_SECTION_TEMPLATE.format(memories=memory_lines)
    else:
        context_section = ""

    return _CLASSIFICATION_PROMPT.format(
        statement=statement,
        context_section=context_section,
    )


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a JSON object from LLM response text.

    The LLM may wrap JSON in markdown fences or add preamble text.
    This function tries several strategies to extract valid JSON.
    """
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON object with regex
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: strip markdown fences and retry
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def _validate_response(data: dict) -> tuple[bool, str]:
    """
    Validate a parsed LLM response dict.

    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Response is not a JSON object."

    # memory_type
    memory_type = data.get("memory_type")
    if not isinstance(memory_type, str):
        return False, f"'memory_type' is missing or not a string: {memory_type!r}"
    if memory_type.upper() not in VALID_MEMORY_TYPES:
        return False, (
            f"'memory_type' must be one of {VALID_MEMORY_TYPES}, got: {memory_type!r}"
        )

    # confidence
    confidence = data.get("confidence")
    if confidence is None:
        return False, "'confidence' field is missing."
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False, f"'confidence' is not a number: {confidence!r}"
    if not (0.0 <= confidence <= 1.0):
        return False, f"'confidence' must be between 0.0 and 1.0, got: {confidence}"

    # reason
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, f"'reason' is missing or empty: {reason!r}"

    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_memory(
    statement: str,
    relevant_memories: Optional[list[dict]] = None,
) -> dict:
    """
    Classify a statement into SEMANTIC, EPISODIC, or PROCEDURAL memory type
    using the local Ollama LLM.

    Parameters
    ----------
    statement : str
        The user input to classify.
    relevant_memories : list[dict], optional
        Previously stored memories retrieved from MongoDB that are relevant
        to the current statement.  Used as context for the LLM.

    Returns
    -------
    dict with keys:
        success       : bool   — True if classification succeeded
        memory_type   : str    — "SEMANTIC" | "EPISODIC" | "PROCEDURAL" | None
        confidence    : float  — 0.0 to 1.0, or None on failure
        reason        : str    — LLM explanation, or error description
        model_used    : str    — actual Ollama model used
        error         : str    — non-empty only when success is False
    """
    if not statement or not statement.strip():
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "Empty statement provided.",
            "model_used": OLLAMA_MODEL,
            "error": "Empty statement — cannot classify.",
        }

    prompt = _build_prompt(statement.strip(), relevant_memories)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},  # low temperature for consistent JSON
        )
        raw_content: str = response["message"]["content"]
        logger.debug("Ollama raw response: %s", raw_content)

    except ollama.ResponseError as exc:
        logger.error("Ollama response error: %s", exc)
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "Ollama returned an error response.",
            "model_used": OLLAMA_MODEL,
            "error": str(exc),
        }
    except Exception as exc:  # connection refused, model not found, etc.
        logger.error("Ollama communication error: %s", exc)
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "Could not communicate with Ollama.",
            "model_used": OLLAMA_MODEL,
            "error": str(exc),
        }

    # Parse JSON from response
    data = _extract_json(raw_content)
    if data is None:
        logger.warning("Could not extract JSON from Ollama response: %s", raw_content)
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "LLM returned non-JSON output.",
            "model_used": OLLAMA_MODEL,
            "error": f"Non-JSON response: {raw_content[:200]}",
        }

    # Validate parsed data
    is_valid, validation_error = _validate_response(data)
    if not is_valid:
        logger.warning("Invalid LLM response structure: %s | data: %s", validation_error, data)
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "LLM returned structurally invalid response.",
            "model_used": OLLAMA_MODEL,
            "error": validation_error,
        }

    return {
        "success": True,
        "memory_type": data["memory_type"].upper(),
        "confidence": round(float(data["confidence"]), 4),
        "reason": data["reason"].strip(),
        "model_used": OLLAMA_MODEL,
        "error": "",
    }


def check_ollama_connection() -> dict:
    """
    Check whether the Ollama service is reachable and the configured model
    is available.

    Returns
    -------
    dict with keys:
        connected     : bool
        model_found   : bool
        model_used    : str
        available_models : list[str]
        error         : str
    """
    try:
        models_response = ollama.list()
        available = [m["model"] for m in models_response.get("models", [])]
        model_found = OLLAMA_MODEL in available
        return {
            "connected": True,
            "model_found": model_found,
            "model_used": OLLAMA_MODEL,
            "available_models": available,
            "error": "" if model_found else f"Model '{OLLAMA_MODEL}' not in {available}",
        }
    except Exception as exc:
        return {
            "connected": False,
            "model_found": False,
            "model_used": OLLAMA_MODEL,
            "available_models": [],
            "error": str(exc),
        }
