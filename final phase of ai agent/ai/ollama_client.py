"""
ai/ollama_client.py
-------------------
Ollama integration for structured memory analysis.

The main memory pipeline asks Ollama to analyze every non-empty user input and
return JSON with:
    memory_type, long_term_beneficial, importance_score, persistence_score,
    usefulness_score, reason

Python validates that JSON before any scoring or MongoDB storage happens.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import ollama

logger = logging.getLogger(__name__)

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:12b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

VALID_MEMORY_TYPES = {"SEMANTIC", "EPISODIC", "PROCEDURAL", "OTHER"}
REQUIRED_ANALYSIS_FIELDS = {
    "memory_type",
    "long_term_beneficial",
    "importance_score",
    "persistence_score",
    "usefulness_score",
    "reason",
}

SYSTEM_MEMORY_ANALYSIS_PROMPT = """\
You are the memory analyzer for a Hybrid Agentic Memory System.

Analyze the meaning of the user's statement. Do not use keyword matching.
Decide based on semantic understanding and context.

Memory means information that may help future conversations with this user.

Memory types:
- SEMANTIC: stable facts, user information, preferences, goals, projects,
  skills, identity, durable knowledge, and ongoing context.
- EPISODIC: events or experiences tied to a particular time, place, or
  situation.
- PROCEDURAL: instructions, workflows, methods, or steps explaining how to do
  something.
- OTHER: questions, commands, temporary remarks, ambiguous text, or statements
  that do not clearly fit the other types.

Assess whether the information is beneficial for long-term memory. This is an
analytical signal only; Python will make the final storage decision.

Scores must be numbers from 0.0 to 1.0:
- importance_score: how valuable the information itself is.
- persistence_score: how stable or durable the information is likely to be.
- usefulness_score: how useful it is likely to be in future conversations.

Return JSON only. Do not include markdown, comments, or extra fields.
Required JSON shape:
{
  "memory_type": "SEMANTIC",
  "long_term_beneficial": true,
  "importance_score": 0.95,
  "persistence_score": 0.99,
  "usefulness_score": 0.95,
  "reason": "Brief reason for the analysis."
}
"""


def _get_client() -> ollama.Client:
    """Build an Ollama client from environment configuration."""
    return ollama.Client(host=OLLAMA_BASE_URL)


def _build_messages(statement: str, recent_context: str | None = None) -> list[dict[str, str]]:
    """Build the chat messages sent to Ollama."""
    user_content = f'User statement:\n"{statement}"'
    if recent_context:
        user_content = (
            "Recent conversation context:\n"
            f"{recent_context.strip()}\n\n"
            f"{user_content}"
        )

    return [
        {"role": "system", "content": SYSTEM_MEMORY_ANALYSIS_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json(text: str | None) -> dict[str, Any] | None:
    """
    Extract a JSON object from an LLM response.

    JSON-only output is required, but this tolerates common wrapper text so one
    bad response does not crash the application.
    """
    if not text:
        return None

    candidates = [text.strip()]
    without_fences = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    candidates.append(without_fences.strip())

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1].strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            try:
                parsed, _ = decoder.raw_decode(candidate)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue

    return None


def _coerce_score(value: Any, field_name: str) -> tuple[float | None, str]:
    """Validate and normalize one score field."""
    if isinstance(value, bool):
        return None, f"'{field_name}' must be numeric, got boolean."
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None, f"'{field_name}' is not numeric: {value!r}"

    if not 0.0 <= score <= 1.0:
        return None, f"'{field_name}' must be between 0.0 and 1.0, got {score}."

    return round(score, 4), ""


def validate_memory_analysis(data: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """
    Validate Ollama's structured memory-analysis JSON.

    Returns (validated_data, error). validated_data is None when invalid.
    """
    if not isinstance(data, dict):
        return None, "Response is not a JSON object."

    missing = sorted(REQUIRED_ANALYSIS_FIELDS - set(data))
    if missing:
        return None, f"Missing required field(s): {', '.join(missing)}."

    memory_type = data.get("memory_type")
    if not isinstance(memory_type, str) or not memory_type.strip():
        return None, "'memory_type' is missing or not a string."
    memory_type = memory_type.strip().upper()
    if memory_type not in VALID_MEMORY_TYPES:
        return None, (
            f"'memory_type' must be one of {sorted(VALID_MEMORY_TYPES)}, "
            f"got {memory_type!r}."
        )

    long_term_beneficial = data.get("long_term_beneficial")
    if not isinstance(long_term_beneficial, bool):
        return None, "'long_term_beneficial' must be a boolean."

    normalized: dict[str, Any] = {
        "memory_type": memory_type,
        "long_term_beneficial": long_term_beneficial,
    }

    for field_name in ("importance_score", "persistence_score", "usefulness_score"):
        score, error = _coerce_score(data.get(field_name), field_name)
        if error:
            return None, error
        normalized[field_name] = score

    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None, "'reason' must be a non-empty string."
    normalized["reason"] = reason.strip()

    return normalized, ""


def _validate_response(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Backward-compatible validation helper used by tests and older imports.

    It now validates the full structured memory-analysis response.
    """
    _, error = validate_memory_analysis(data)
    return error == "", error


def _read_response_content(response: Any) -> str:
    """Read message.content from dict-like or object-like Ollama responses."""
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(getattr(message, "content", "") or "")

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _chat_json(messages: list[dict[str, str]]) -> Any:
    """
    Call Ollama using JSON response mode when supported.

    Older Ollama clients may not accept the format argument, so a TypeError
    falls back to a normal chat call while preserving the JSON-only prompt.
    """
    client = _get_client()
    try:
        return client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={"temperature": 0.1},
        )
    except TypeError:
        return client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1},
        )


def analyze_memory(statement: str, recent_context: str | None = None) -> dict[str, Any]:
    """
    Analyze a user statement with Ollama and return validated structured data.

    On failure, success is False and no fabricated memory fields are returned.
    """
    text = (statement or "").strip()
    if not text:
        return {
            "success": False,
            "model_used": OLLAMA_MODEL,
            "error": "Empty input - nothing to analyze.",
            "reason": "Empty input - nothing to analyze.",
        }

    messages = _build_messages(text, recent_context)

    try:
        response = _chat_json(messages)
        raw_content = _read_response_content(response)
        logger.debug("Ollama raw memory analysis response: %s", raw_content)
    except Exception as exc:
        logger.error("Ollama memory analysis failed: %s", exc)
        return {
            "success": False,
            "model_used": OLLAMA_MODEL,
            "error": str(exc),
            "reason": "Could not communicate with Ollama.",
        }

    data = _extract_json(raw_content)
    if data is None:
        return {
            "success": False,
            "model_used": OLLAMA_MODEL,
            "error": f"Malformed or non-JSON Ollama response: {raw_content[:200]}",
            "reason": "Ollama returned malformed or non-JSON output.",
        }

    validated, validation_error = validate_memory_analysis(data)
    if validation_error:
        return {
            "success": False,
            "model_used": OLLAMA_MODEL,
            "error": validation_error,
            "reason": "Ollama returned invalid memory analysis JSON.",
        }

    return {
        "success": True,
        **validated,
        "model_used": OLLAMA_MODEL,
        "error": "",
    }


def classify_memory(
    statement: str,
    relevant_memories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compatibility wrapper for the old classification-only API.

    The main pipeline no longer uses this function. It delegates to
    analyze_memory so older callers still route through Ollama first.
    """
    context = None
    if relevant_memories:
        context = "\n".join(
            f"- [{memory.get('memory_type', 'OTHER')}] "
            f"{memory.get('memory_text') or memory.get('memory', '')}"
            for memory in relevant_memories[:5]
        )

    analysis = analyze_memory(statement, recent_context=context)
    if not analysis["success"]:
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": analysis.get("reason", ""),
            "model_used": OLLAMA_MODEL,
            "error": analysis.get("error", ""),
        }

    confidence = round(
        (
            analysis["importance_score"]
            + analysis["persistence_score"]
            + analysis["usefulness_score"]
        ) / 3,
        4,
    )
    return {
        "success": True,
        "memory_type": analysis["memory_type"],
        "confidence": confidence,
        "reason": analysis["reason"],
        "model_used": OLLAMA_MODEL,
        "error": "",
    }


def check_ollama_connection() -> dict[str, Any]:
    """
    Check whether Ollama is reachable and whether the configured model exists.
    """
    try:
        models_response = _get_client().list()
        if isinstance(models_response, dict):
            models = models_response.get("models", [])
        else:
            models = getattr(models_response, "models", [])

        available: list[str] = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                available.append(str(name))

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
