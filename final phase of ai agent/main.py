"""
main.py
────────
Entry point for the Hybrid Agentic Memory System.

Pipeline (per user input):
    1. Context Analysis      — is this relevant?
    2. Importance Analysis   — is this important?
    3. Persistence Analysis  — will this remain useful?
    4. Long-Term Decision    — should it be stored long-term?
       ├── NO  → report decision, skip storage
       └── YES → classify memory type via Ollama LLM → store in MongoDB

Run:
    python main.py
"""

import logging
import os
import sys

# ── Load .env file if python-dotenv is available (optional) ───────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use system env vars as-is

# Configure UTF-8 encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Configure logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Show WARNING+ to keep console clean
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────────
from context_analysis import analyze_context
from importance_analysis import analyze_importance
from persistence_analysis import analyze_persistence
from longterm_memory import decide_longterm
from memory_types.memory_type_analyzer import analyze_and_store
from ai.ollama_client import check_ollama_connection
from database.mongodb import check_mongodb_connection

# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def _print_startup_status() -> None:
    """Print a brief status of Ollama and MongoDB connections at startup."""
    print("\n" + "═" * 60)
    print("  Hybrid Agentic Memory System")
    print("═" * 60)

    # Ollama
    ollama_status = check_ollama_connection()
    if ollama_status["connected"] and ollama_status["model_found"]:
        print(f"  ✅ Ollama   : connected  | model: {ollama_status['model_used']}")
    elif ollama_status["connected"]:
        print(f"  ⚠️  Ollama   : connected  | model '{ollama_status['model_used']}' NOT found")
        print(f"     Available: {ollama_status['available_models']}")
    else:
        print(f"  ❌ Ollama   : UNAVAILABLE — {ollama_status['error']}")

    # MongoDB
    mongo_status = check_mongodb_connection()
    if mongo_status["connected"]:
        print(f"  ✅ MongoDB  : connected  | db: {mongo_status['db_name']}")
    else:
        print(f"  ❌ MongoDB  : UNAVAILABLE — {mongo_status['error']}")

    print("═" * 60)
    print("  Type your statement and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Per-input pipeline
# ---------------------------------------------------------------------------

def process_input(
    user_input: str,
    recent_context: str | None = None,
) -> dict:
    """
    Run the full memory pipeline on a single user input.

    Parameters
    ----------
    user_input     : str — the raw user statement
    recent_context : str, optional — last few lines of conversation context

    Returns
    -------
    dict — full pipeline result containing all scores and the final decision
    """
    text = (user_input or "").strip()

    # ── Guard: empty input ────────────────────────────────────────────────
    if not text:
        return {
            "input": text,
            "pipeline_complete": False,
            "reason": "Empty input — nothing to process.",
        }

    result: dict = {"input": text}

    # ── Step 1: Context Analysis ──────────────────────────────────────────
    ctx = analyze_context(text, recent_context)
    result["context"] = ctx
    context_score: float = ctx["context_score"]

    # ── Step 2: Importance Analysis ───────────────────────────────────────
    imp = analyze_importance(text)
    result["importance"] = imp
    importance_score: float = imp["importance_score"]

    # ── Step 3: Persistence Analysis ─────────────────────────────────────
    per = analyze_persistence(text)
    result["persistence"] = per
    persistence_score: float = per["persistence_score"]

    # ── Step 4: Long-Term Memory Decision ─────────────────────────────────
    lt = decide_longterm(context_score, importance_score, persistence_score)
    result["longterm_decision"] = lt

    if not lt["is_longterm"]:
        result["pipeline_complete"] = True
        result["stored"] = False
        result["reason"] = "Not stored: did not pass the long-term memory threshold."
        return result

    # ── Step 5: Memory Type Classification + MongoDB Storage ─────────────
    classification = analyze_and_store(
        memory=text,
        context_score=context_score,
        importance_score=importance_score,
        persistence_score=persistence_score,
    )
    result["classification"] = classification
    result["pipeline_complete"] = True
    result["stored"] = classification["success"] and classification["inserted_id"] is not None

    if classification["success"]:
        result["reason"] = (
            f"Stored as {classification['memory_type']} memory "
            f"(confidence: {classification['confidence']:.2f})."
        )
    else:
        result["reason"] = (
            f"Classification failed: {classification['error']}"
        )

    return result


# ---------------------------------------------------------------------------
# Pretty-print pipeline result
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    """Display the pipeline result in a readable format."""
    print()
    print("─" * 60)

    if not result.get("pipeline_complete"):
        print(f"  ⚠️  {result.get('reason', 'Unknown error.')}")
        print("─" * 60)
        return

    # Scores
    ctx = result.get("context", {})
    imp = result.get("importance", {})
    per = result.get("persistence", {})
    lt  = result.get("longterm_decision", {})

    print(f"  📊 Scores:")
    print(f"     Context     : {ctx.get('context_score', 'N/A'):.4f}  "
          f"({'✅' if ctx.get('is_relevant') else '❌'})")
    print(f"     Importance  : {imp.get('importance_score', 'N/A'):.4f}  "
          f"({'✅' if imp.get('is_important') else '❌'})")
    print(f"     Persistence : {per.get('persistence_score', 'N/A'):.4f}  "
          f"({'✅' if per.get('is_persistent') else '❌'})")
    print(f"     Combined    : {lt.get('combined_score', 'N/A'):.4f}")

    # Long-term decision
    is_longterm = lt.get("is_longterm", False)
    print(f"\n  🧠 Long-Term Decision : {'YES — classifying' if is_longterm else 'NO — not storing'}")

    if not is_longterm:
        print(f"     Reason: {lt.get('reason', '')}")
        print("─" * 60)
        return

    # Classification result
    clf = result.get("classification", {})
    if clf.get("success"):
        mem_type = clf.get("memory_type", "?")
        confidence = clf.get("confidence", 0.0)
        reason = clf.get("reason", "")
        inserted_id = clf.get("inserted_id")
        ctx_used = clf.get("relevant_memories_used", 0)

        type_icons = {"SEMANTIC": "📚", "EPISODIC": "📅", "PROCEDURAL": "⚙️"}
        icon = type_icons.get(mem_type, "🧩")

        print(f"\n  {icon} Memory Type    : {mem_type}")
        print(f"     Confidence   : {confidence:.4f}")
        print(f"     Reason       : {reason}")
        if ctx_used:
            print(f"     Context used : {ctx_used} previous memor{'y' if ctx_used == 1 else 'ies'}")
        if inserted_id:
            print(f"\n  ✅ Stored in MongoDB  (id: {inserted_id})")
        else:
            print(f"\n  ⚠️  Classified but NOT stored (MongoDB unavailable).")
    else:
        print(f"\n  ❌ Classification failed: {clf.get('error', 'Unknown error')}")

    print("─" * 60)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    _print_startup_status()

    recent_context: str | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            sys.exit(0)

        if not user_input:
            print("  (Empty input — please type something.)")
            continue

        result = process_input(user_input, recent_context)
        _print_result(result)

        # Update rolling context (last non-trivial input)
        recent_context = user_input


if __name__ == "__main__":
    main()
