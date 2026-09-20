"""
memory_types/memory_type_analyzer.py
──────────────────────────────────────
Main coordinator for memory type classification.

This module is the bridge between the Long-Term Memory decision and MongoDB
storage.  It is called ONLY when the long-term decision is True.

Responsibilities:
    1. Retrieve relevant previous memories from MongoDB (for context).
    2. Call the Ollama LLM via ollama_client to classify the statement.
    3. Validate the LLM output.
    4. Route to the appropriate memory-type module (semantic / episodic /
       procedural) to build a structured record.
    5. Insert the record into MongoDB.
    6. Return a full result dict to the caller.

This module does NOT duplicate classification logic — it delegates:
    - LLM reasoning  → ai.ollama_client
    - Record building → memory_types.{semantic,episodic,procedural}_memory
    - Persistence     → database.mongodb
"""

import logging
from typing import Optional

from ai.ollama_client import classify_memory
from database.mongodb import MemoryDatabase
from memory_types import semantic_memory, episodic_memory, procedural_memory

logger = logging.getLogger(__name__)

# Map memory_type string to the appropriate record-builder module
_TYPE_MODULE_MAP = {
    "SEMANTIC": semantic_memory,
    "EPISODIC": episodic_memory,
    "PROCEDURAL": procedural_memory,
}


def analyze_and_store(
    memory: str,
    context_score: float,
    importance_score: float,
    persistence_score: float,
    db: Optional[MemoryDatabase] = None,
) -> dict:
    """
    Classify a statement and store it in MongoDB.

    This function is called after the Long-Term Memory decision returns True.

    Parameters
    ----------
    memory            : str   — the user's statement to classify and store
    context_score     : float — score from context_analysis
    importance_score  : float — score from importance_analysis
    persistence_score : float — score from persistence_analysis
    db                : MemoryDatabase, optional
                        An already-connected MemoryDatabase instance.
                        If None, a new connection is created (and closed
                        after the operation).

    Returns
    -------
    dict with keys:
        success         : bool
        memory_type     : str | None
        confidence      : float | None
        reason          : str
        inserted_id     : str | None  — MongoDB document _id
        relevant_memories_used : int  — number of context memories provided to LLM
        error           : str         — non-empty only on failure
    """
    statement = (memory or "").strip()
    if not statement:
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "Empty statement — cannot classify.",
            "inserted_id": None,
            "relevant_memories_used": 0,
            "error": "Empty statement.",
        }

    # ── Step 1: Retrieve relevant previous memories ────────────────────────
    relevant_memories: list[dict] = []
    own_db = False

    if db is None:
        own_db = True
        db = MemoryDatabase()
        try:
            db.connect()
        except Exception as exc:
            logger.warning(
                "MongoDB unavailable for context retrieval: %s", exc
            )
            db = None

    if db is not None:
        try:
            relevant_memories = db.get_relevant_memories(statement, limit=5)
            logger.debug(
                "Retrieved %d relevant memories for context.", len(relevant_memories)
            )
        except Exception as exc:
            logger.warning("Could not retrieve relevant memories: %s", exc)

    # ── Step 2: LLM classification ────────────────────────────────────────
    llm_result = classify_memory(statement, relevant_memories or None)

    if not llm_result["success"]:
        if own_db and db:
            db.close()
        return {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": llm_result["reason"],
            "inserted_id": None,
            "relevant_memories_used": len(relevant_memories),
            "error": llm_result["error"],
        }

    memory_type: str = llm_result["memory_type"]
    confidence: float = llm_result["confidence"]
    reason: str = llm_result["reason"]

    # ── Step 3: Build structured record ───────────────────────────────────
    type_module = _TYPE_MODULE_MAP.get(memory_type)
    if type_module is None:
        # Defensive: should never happen after validation
        if own_db and db:
            db.close()
        return {
            "success": False,
            "memory_type": memory_type,
            "confidence": confidence,
            "reason": reason,
            "inserted_id": None,
            "relevant_memories_used": len(relevant_memories),
            "error": f"Unknown memory_type after validation: {memory_type!r}",
        }

    record = type_module.build_record(
        memory=statement,
        context_score=context_score,
        importance_score=importance_score,
        persistence_score=persistence_score,
        confidence_score=confidence,
        reason=reason,
    )

    # ── Step 4: Store in MongoDB ───────────────────────────────────────────
    inserted_id: Optional[str] = None

    if db is not None:
        try:
            inserted_id = db.insert_memory(
                memory=record["memory"],
                memory_type=record["memory_type"],
                context_score=record["context_score"],
                importance_score=record["importance_score"],
                persistence_score=record["persistence_score"],
                confidence_score=record["confidence_score"],
                reason=record["reason"],
            )
        except Exception as exc:
            logger.error("Failed to store memory in MongoDB: %s", exc)
            if own_db:
                db.close()
            return {
                "success": False,
                "memory_type": memory_type,
                "confidence": confidence,
                "reason": reason,
                "inserted_id": None,
                "relevant_memories_used": len(relevant_memories),
                "error": f"MongoDB insertion failed: {exc}",
            }
    else:
        logger.warning("MongoDB unavailable — memory classified but NOT stored.")

    if own_db and db:
        db.close()

    return {
        "success": True,
        "memory_type": memory_type,
        "confidence": confidence,
        "reason": reason,
        "inserted_id": inserted_id,
        "relevant_memories_used": len(relevant_memories),
        "error": "",
    }
