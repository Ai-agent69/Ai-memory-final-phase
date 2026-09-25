"""
main.py
-------
Entry point for the Hybrid Agentic Memory System.

Required pipeline per user input:
    User Input
        -> Ollama structured memory analysis JSON
        -> Temporary scope goes directly to Trash
        -> Python validation
        -> Python final score for potential long-term memories only
        -> Python threshold decision: LONG_TERM or TRASH
        -> MongoDB storage only for validated LONG_TERM memories

Run:
    python main.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

from ai.ollama_client import analyze_memory, check_ollama_connection
from database.mongodb import MemoryDatabase, check_mongodb_connection
from memory_evaluator import (
    LONG_TERM_THRESHOLD,
    build_memory_document,
    make_memory_decision,
)
from trash_box import record_trash


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def _print_startup_status() -> None:
    """Print a brief status of Ollama and MongoDB connections at startup."""
    print("\n" + "=" * 60)
    print("Hybrid Agentic Memory System")
    print("=" * 60)

    ollama_status = check_ollama_connection()
    if ollama_status["connected"] and ollama_status["model_found"]:
        print(f"Ollama : CONNECTED | model: {ollama_status['model_used']}")
    elif ollama_status["connected"]:
        print(f"Ollama : CONNECTED | model '{ollama_status['model_used']}' not found")
        print(f"Available models: {ollama_status['available_models']}")
    else:
        print(f"Ollama : UNAVAILABLE | {ollama_status['error']}")

    mongo_status = check_mongodb_connection()
    if mongo_status["connected"]:
        print(f"MongoDB: CONNECTED | db: {mongo_status['db_name']}")
    else:
        print(f"MongoDB: UNAVAILABLE | {mongo_status['error']}")

    print("=" * 60)
    print("Type a statement and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")


def _store_long_term_memory(memory_text: str, analysis: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Store one approved memory document in MongoDB."""
    document = build_memory_document(memory_text, analysis, decision)
    db = MemoryDatabase()

    try:
        db.connect()
    except Exception as exc:
        logger.warning("MongoDB unavailable: %s", exc)
        return {
            "status": "UNAVAILABLE",
            "stored": False,
            "inserted_id": None,
            "error": str(exc),
            "document": document,
        }

    try:
        inserted_id = db.insert_analyzed_memory(document)
        return {
            "status": "STORED",
            "stored": True,
            "inserted_id": inserted_id,
            "error": "",
            "document": document,
        }
    except Exception as exc:
        logger.error("MongoDB storage failed: %s", exc)
        return {
            "status": "FAILED",
            "stored": False,
            "inserted_id": None,
            "error": str(exc),
            "document": document,
        }
    finally:
        db.close()


def process_input(
    user_input: str,
    recent_context: str | None = None,
) -> dict[str, Any]:
    """
    Run the Ollama-first memory pipeline on a single user input.

    Ollama analyzes every non-empty input. Temporary scope goes directly to
    Trash. Python validation, scoring, and the threshold decision happen before
    MongoDB is touched for potential long-term memories.
    """
    text = (user_input or "").strip()
    if not text:
        return {
            "input": text,
            "pipeline_complete": False,
            "analysis_success": False,
            "stored": False,
            "mongodb_status": "NOT_ATTEMPTED",
            "reason": "Empty input - nothing to process.",
        }

    result: dict[str, Any] = {
        "input": text,
        "pipeline_complete": False,
        "analysis_success": False,
        "stored": False,
        "mongodb_status": "NOT_ATTEMPTED",
    }

    analysis = analyze_memory(text, recent_context=recent_context)
    result["analysis"] = analysis

    if not analysis["success"]:
        result.update(
            {
                "pipeline_complete": False,
                "reason": analysis.get("reason", "Memory analysis failed."),
                "error": analysis.get("error", ""),
            }
        )
        return result

    try:
        decision = make_memory_decision(analysis, threshold=LONG_TERM_THRESHOLD)
    except (KeyError, TypeError, ValueError) as exc:
        result.update(
            {
                "pipeline_complete": False,
                "reason": "Validated analysis could not be evaluated.",
                "error": str(exc),
            }
        )
        return result

    result.update(
        {
            "pipeline_complete": True,
            "analysis_success": True,
            "memory_type": analysis["memory_type"],
            "memory_scope": analysis["memory_scope"],
            "long_term_beneficial": analysis["long_term_beneficial"],
            "final_score": decision["final_score"],
            "threshold": decision["threshold"],
            "decision": decision["decision"],
            "scoring_skipped": decision["scoring_skipped"],
            "reason": analysis["reason"],
            "decision_reason": decision["reason"],
        }
    )

    if not decision["scoring_skipped"]:
        result.update(
            {
                "importance_score": analysis["importance_score"],
                "persistence_score": analysis["persistence_score"],
                "usefulness_score": analysis["usefulness_score"],
            }
        )

    if not decision["is_longterm"]:
        trash_entry = record_trash(
            input_text=text,
            memory_type=analysis["memory_type"],
            memory_scope=analysis["memory_scope"],
            reason=analysis["reason"],
            final_score=decision["final_score"],
            threshold=decision["threshold"],
            scoring_skipped=decision["scoring_skipped"],
        )
        result.update(
            {
                "stored": False,
                "mongodb_status": "NOT STORED",
                "storage": "NOT STORED",
                "storage_error": "",
                "trash_entry": trash_entry,
            }
        )
        return result

    storage = _store_long_term_memory(text, analysis, decision)
    result["storage"] = storage
    result["stored"] = storage["stored"]
    result["mongodb_status"] = storage["status"]
    result["storage_error"] = storage["error"]

    return result


def _print_result(result: dict[str, Any]) -> None:
    """Display the memory analysis in a clear, test-friendly format."""
    print()
    print("=" * 40)
    print("MEMORY ANALYSIS")
    print("=" * 40)
    print("Input:")
    print(result.get("input", ""))
    print()

    if not result.get("analysis_success"):
        print("Memory Type:")
        print("N/A")
        print()
        print("Decision:")
        print("NOT ANALYZED")
        print()
        print("MongoDB:")
        print(result.get("mongodb_status", "NOT_ATTEMPTED"))
        print()
        print("Reason:")
        print(result.get("reason", "Analysis failed."))
        error = result.get("error")
        if error:
            print()
            print("Error:")
            print(error)
        print("=" * 40)
        return

    print("Memory Type:")
    print(result["memory_type"])
    print()
    print("Memory Scope:")
    print(result["memory_scope"])
    print()

    if result.get("scoring_skipped"):
        print("Decision:")
        print(result["decision"])
        print()
        print("Scoring:")
        print("SKIPPED")
        print()
        print("Storage:")
        print("NOT STORED")
        print()
        print("Reason:")
        print(result["reason"])
        print("=" * 40)
        return

    print("Importance Score:")
    print(f"{result['importance_score']:.4f}")
    print()
    print("Persistence Score:")
    print(f"{result['persistence_score']:.4f}")
    print()
    print("Usefulness Score:")
    print(f"{result['usefulness_score']:.4f}")
    print()
    print("Final Score:")
    print(f"{result['final_score']:.4f}")
    print()
    print("Threshold:")
    print(f"{result['threshold']:.2f}")
    print()
    print("Decision:")
    print(result["decision"])
    print()
    if result["decision"] == "LONG_TERM":
        print("MongoDB:")
        print(result["mongodb_status"])
        if result.get("storage_error"):
            print(result["storage_error"])
    else:
        print("Storage:")
        print("NOT STORED")
    print()
    print("Reason:")
    print(result["reason"])
    print("=" * 40)


def main() -> None:
    _print_startup_status()

    recent_context: str | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            sys.exit(0)

        if not user_input:
            print("(Empty input - please type something.)")
            continue

        result = process_input(user_input, recent_context)
        _print_result(result)
        recent_context = user_input


if __name__ == "__main__":
    main()
