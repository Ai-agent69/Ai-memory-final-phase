"""
database/mongodb.py
────────────────────
MongoDB persistence layer for the Hybrid Agentic Memory System.

Responsibilities:
    - Connect to MongoDB (URI from MONGODB_URI env var).
    - Insert, retrieve, update, and delete memory documents.
    - Query by memory_type.
    - Retrieve relevant previous memories for LLM context.

Database  : agentic_memory  (configurable via MONGODB_DB env var)
Collection: memories

Document schema:
    {
        "memory"           : str,
        "memory_type"      : "SEMANTIC" | "EPISODIC" | "PROCEDURAL",
        "context_score"    : float,
        "importance_score" : float,
        "persistence_score": float,
        "confidence_score" : float,
        "reason"           : str,
        "created_at"       : str  (ISO-8601)
    }

Environment variables:
    MONGODB_URI  — connection URI (default: mongodb://localhost:27017/)
    MONGODB_DB   — database name  (default: agentic_memory)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from pymongo import MongoClient, TEXT, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB: str = os.getenv("MONGODB_DB", "agentic_memory")
COLLECTION_NAME: str = "memories"

VALID_MEMORY_TYPES = {"SEMANTIC", "EPISODIC", "PROCEDURAL"}


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

class MemoryDatabase:
    """
    Manages the connection to MongoDB and all memory CRUD operations.

    Usage (preferred):
        db = MemoryDatabase()
        db.connect()
        db.insert_memory(...)
        db.close()

    Or use as a context manager:
        with MemoryDatabase() as db:
            db.insert_memory(...)
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        self._uri = uri or MONGODB_URI
        self._db_name = db_name or MONGODB_DB
        self._client: Optional[MongoClient] = None
        self._collection: Optional[Collection] = None

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> "MemoryDatabase":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Connect to MongoDB and initialise the collection.

        Raises
        ------
        ConnectionFailure
            If MongoDB is unreachable.
        """
        try:
            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=5000,  # 5s timeout for connection
            )
            # Ping to verify connection
            self._client.admin.command("ping")
            db = self._client[self._db_name]
            self._collection = db[COLLECTION_NAME]
            self._ensure_indexes()
            logger.info("Connected to MongoDB: %s / %s", self._uri, self._db_name)
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.error("MongoDB connection failed: %s", exc)
            raise

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._collection = None
            logger.info("MongoDB connection closed.")

    @property
    def is_connected(self) -> bool:
        return self._collection is not None

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient querying."""
        try:
            # Index on memory_type for filtered queries
            self._collection.create_index("memory_type")
            # Index on created_at for chronological retrieval
            self._collection.create_index([("created_at", DESCENDING)])
            # Text index on memory field for relevant-memory retrieval
            self._collection.create_index([("memory", TEXT)], name="memory_text_index")
            logger.debug("MongoDB indexes ensured.")
        except OperationFailure as exc:
            # Non-fatal — log and continue
            logger.warning("Could not create indexes: %s", exc)

    def _require_connection(self) -> None:
        if not self.is_connected:
            raise RuntimeError(
                "Not connected to MongoDB. Call connect() first."
            )

    # ── Insert ─────────────────────────────────────────────────────────────

    def insert_memory(
        self,
        memory: str,
        memory_type: str,
        context_score: float,
        importance_score: float,
        persistence_score: float,
        confidence_score: float,
        reason: str,
    ) -> str:
        """
        Insert a new memory document.

        Parameters
        ----------
        memory            : str   — the original user statement
        memory_type       : str   — SEMANTIC | EPISODIC | PROCEDURAL
        context_score     : float
        importance_score  : float
        persistence_score : float
        confidence_score  : float — LLM classification confidence
        reason            : str   — LLM explanation

        Returns
        -------
        str — the inserted document's _id as a string

        Raises
        ------
        ValueError   — invalid memory_type
        RuntimeError — not connected
        """
        self._require_connection()

        memory_type_upper = (memory_type or "").strip().upper()
        if memory_type_upper not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type: {memory_type!r}. "
                f"Must be one of {VALID_MEMORY_TYPES}."
            )

        document = {
            "memory": memory.strip(),
            "memory_type": memory_type_upper,
            "context_score": round(float(context_score), 4),
            "importance_score": round(float(importance_score), 4),
            "persistence_score": round(float(persistence_score), 4),
            "confidence_score": round(float(confidence_score), 4),
            "reason": reason.strip(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = self._collection.insert_one(document)
            inserted_id = str(result.inserted_id)
            logger.info(
                "Memory inserted: id=%s type=%s", inserted_id, memory_type_upper
            )
            return inserted_id
        except Exception as exc:
            logger.error("Failed to insert memory: %s", exc)
            raise

    # ── Retrieve ───────────────────────────────────────────────────────────

    def get_memory_by_id(self, memory_id: str) -> Optional[dict]:
        """Retrieve a single memory document by its string _id."""
        self._require_connection()
        try:
            doc = self._collection.find_one({"_id": ObjectId(memory_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception as exc:
            logger.error("Error retrieving memory %s: %s", memory_id, exc)
            return None

    def get_memories_by_type(
        self,
        memory_type: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Retrieve memories filtered by memory_type, most recent first.

        Parameters
        ----------
        memory_type : str — SEMANTIC | EPISODIC | PROCEDURAL
        limit       : int — maximum number of results (default 50)
        """
        self._require_connection()
        memory_type_upper = (memory_type or "").strip().upper()
        if memory_type_upper not in VALID_MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type: {memory_type!r}")

        try:
            cursor = (
                self._collection
                .find({"memory_type": memory_type_upper})
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            docs = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
            return docs
        except Exception as exc:
            logger.error("Error querying by type %s: %s", memory_type_upper, exc)
            return []

    def get_all_memories(self, limit: int = 100) -> list[dict]:
        """Retrieve all memories, most recent first."""
        self._require_connection()
        try:
            cursor = (
                self._collection
                .find()
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            docs = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
            return docs
        except Exception as exc:
            logger.error("Error retrieving all memories: %s", exc)
            return []

    def get_relevant_memories(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most relevant previous memories for a given query
        string, using MongoDB's full-text search index.

        Only a limited number of memories are returned to avoid sending
        excessive context to the LLM.

        Parameters
        ----------
        query : str — the current user statement
        limit : int — maximum memories to return (default 5)

        Returns
        -------
        list[dict] — matching memory documents (empty list on failure)
        """
        self._require_connection()

        if not query or not query.strip():
            return []

        try:
            # MongoDB text search
            cursor = (
                self._collection
                .find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}},
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(limit)
            )
            docs = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)

            if docs:
                logger.debug("Found %d relevant memories for query.", len(docs))
                return docs

        except Exception as exc:
            logger.warning("Text search failed, falling back to recent: %s", exc)

        # Fallback: return most recent memories if text search fails
        try:
            cursor = (
                self._collection
                .find()
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            docs = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
            return docs
        except Exception as exc:
            logger.error("Fallback retrieval failed: %s", exc)
            return []

    # ── Delete ─────────────────────────────────────────────────────────────

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory document by its string _id.

        Returns True if the document was deleted, False otherwise.
        """
        self._require_connection()
        try:
            result = self._collection.delete_one({"_id": ObjectId(memory_id)})
            deleted = result.deleted_count > 0
            if deleted:
                logger.info("Memory deleted: id=%s", memory_id)
            else:
                logger.warning("Memory not found for deletion: id=%s", memory_id)
            return deleted
        except Exception as exc:
            logger.error("Error deleting memory %s: %s", memory_id, exc)
            return False

    def delete_all_memories(self) -> int:
        """Delete all memories. Returns the count of deleted documents."""
        self._require_connection()
        try:
            result = self._collection.delete_many({})
            logger.info("Deleted %d memories.", result.deleted_count)
            return result.deleted_count
        except Exception as exc:
            logger.error("Error deleting all memories: %s", exc)
            return 0

    # ── Count ──────────────────────────────────────────────────────────────

    def count_memories(self, memory_type: Optional[str] = None) -> int:
        """Count stored memories, optionally filtered by type."""
        self._require_connection()
        query = {}
        if memory_type:
            query["memory_type"] = memory_type.upper()
        try:
            return self._collection.count_documents(query)
        except Exception as exc:
            logger.error("Error counting memories: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def check_mongodb_connection(
    uri: Optional[str] = None,
    db_name: Optional[str] = None,
) -> dict:
    """
    Check whether MongoDB is reachable.

    Returns
    -------
    dict with keys:
        connected : bool
        uri       : str
        db_name   : str
        error     : str
    """
    _uri = uri or MONGODB_URI
    _db = db_name or MONGODB_DB
    try:
        client = MongoClient(_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        client.close()
        return {"connected": True, "uri": _uri, "db_name": _db, "error": ""}
    except Exception as exc:
        return {"connected": False, "uri": _uri, "db_name": _db, "error": str(exc)}
