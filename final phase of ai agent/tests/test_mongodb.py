"""tests/test_mongodb.py — Unit tests for database/mongodb.py (mocked pymongo)"""

import pytest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database.mongodb import MemoryDatabase, check_mongodb_connection, VALID_MEMORY_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_collection():
    """Return a mock pymongo Collection."""
    col = MagicMock()
    col.insert_one.return_value = MagicMock(inserted_id="507f1f77bcf86cd799439011")
    col.find_one.return_value = None
    col.find.return_value = iter([])
    col.count_documents.return_value = 0
    col.delete_one.return_value = MagicMock(deleted_count=1)
    col.delete_many.return_value = MagicMock(deleted_count=0)
    col.create_index.return_value = None
    return col


def _make_connected_db():
    """Return a MemoryDatabase with a mocked MongoDB connection."""
    db = MemoryDatabase()
    mock_col = _make_mock_collection()
    db._collection = mock_col
    db._client = MagicMock()
    return db, mock_col


# ---------------------------------------------------------------------------
# MemoryDatabase.insert_memory
# ---------------------------------------------------------------------------

class TestInsertMemory:

    def test_insert_valid_semantic(self):
        db, col = _make_connected_db()
        inserted_id = db.insert_memory(
            memory="MongoDB is a document database.",
            memory_type="SEMANTIC",
            context_score=0.8,
            importance_score=0.9,
            persistence_score=0.85,
            confidence_score=0.92,
            reason="factual",
        )
        assert inserted_id is not None
        assert col.insert_one.called

    def test_insert_valid_episodic(self):
        db, col = _make_connected_db()
        db.insert_memory(
            memory="Yesterday we decided to use MongoDB.",
            memory_type="EPISODIC",
            context_score=0.7,
            importance_score=0.75,
            persistence_score=0.6,
            confidence_score=0.88,
            reason="event",
        )
        assert col.insert_one.called

    def test_insert_valid_procedural(self):
        db, col = _make_connected_db()
        db.insert_memory(
            memory="First connect, then query.",
            memory_type="PROCEDURAL",
            context_score=0.75,
            importance_score=0.8,
            persistence_score=0.9,
            confidence_score=0.85,
            reason="procedure",
        )
        assert col.insert_one.called

    def test_insert_invalid_memory_type_raises(self):
        db, _ = _make_connected_db()
        with pytest.raises(ValueError, match="Invalid memory_type"):
            db.insert_memory(
                memory="test",
                memory_type="INVALID",
                context_score=0.5,
                importance_score=0.5,
                persistence_score=0.5,
                confidence_score=0.5,
                reason="test",
            )

    def test_insert_requires_connection(self):
        db = MemoryDatabase()  # not connected
        with pytest.raises(RuntimeError, match="Not connected"):
            db.insert_memory(
                memory="test",
                memory_type="SEMANTIC",
                context_score=0.5,
                importance_score=0.5,
                persistence_score=0.5,
                confidence_score=0.5,
                reason="test",
            )

    def test_document_has_all_required_fields(self):
        db, col = _make_connected_db()
        db.insert_memory(
            memory="Python is a language.",
            memory_type="SEMANTIC",
            context_score=0.8,
            importance_score=0.85,
            persistence_score=0.9,
            confidence_score=0.95,
            reason="factual",
        )
        call_args = col.insert_one.call_args[0][0]
        required_fields = {
            "memory", "memory_type", "context_score",
            "importance_score", "persistence_score",
            "confidence_score", "reason", "created_at",
        }
        assert required_fields.issubset(call_args.keys())

    def test_memory_type_stored_uppercase(self):
        db, col = _make_connected_db()
        db.insert_memory(
            memory="Test",
            memory_type="semantic",  # lowercase input
            context_score=0.5,
            importance_score=0.5,
            persistence_score=0.5,
            confidence_score=0.5,
            reason="test",
        )
        call_args = col.insert_one.call_args[0][0]
        assert call_args["memory_type"] == "SEMANTIC"


class TestInsertAnalyzedMemory:

    def _document(self):
        return {
            "memory_text": "Python is a programming language.",
            "memory_type": "SEMANTIC",
            "long_term_beneficial": True,
            "importance_score": 0.9,
            "persistence_score": 0.95,
            "usefulness_score": 0.85,
            "final_score": 0.9025,
            "decision": "LONG_TERM",
            "reason": "Stable knowledge.",
        }

    def test_insert_valid_analyzed_memory(self):
        db, col = _make_connected_db()
        inserted_id = db.insert_analyzed_memory(self._document())
        assert inserted_id is not None
        document = col.insert_one.call_args[0][0]
        assert document["memory_text"] == "Python is a programming language."
        assert document["memory"] == document["memory_text"]
        assert document["decision"] == "LONG_TERM"

    def test_insert_rejects_temporary_decision(self):
        db, _ = _make_connected_db()
        document = self._document()
        document["decision"] = "TEMPORARY"
        with pytest.raises(ValueError, match="Only LONG_TERM"):
            db.insert_analyzed_memory(document)

    def test_insert_rejects_invalid_score(self):
        db, _ = _make_connected_db()
        document = self._document()
        document["usefulness_score"] = 1.5
        with pytest.raises(ValueError, match="usefulness_score"):
            db.insert_analyzed_memory(document)

    def test_insert_rejects_non_boolean_beneficial_flag(self):
        db, _ = _make_connected_db()
        document = self._document()
        document["long_term_beneficial"] = "true"
        with pytest.raises(ValueError, match="long_term_beneficial"):
            db.insert_analyzed_memory(document)


# ---------------------------------------------------------------------------
# MemoryDatabase.get_memories_by_type
# ---------------------------------------------------------------------------

class TestGetMemoriesByType:

    def test_get_semantic_memories(self):
        db, col = _make_connected_db()
        mock_doc = {
            "_id": MagicMock(__str__=lambda s: "507f1f77bcf86cd799439011"),
            "memory": "MongoDB is a DB.",
            "memory_type": "SEMANTIC",
        }
        # Build the chain: col.find(...).sort(...).limit(...) → iterable
        mock_limit = MagicMock()
        mock_limit.__iter__ = MagicMock(return_value=iter([mock_doc]))
        col.find.return_value = MagicMock()
        col.find.return_value.sort.return_value = MagicMock()
        col.find.return_value.sort.return_value.limit.return_value = mock_limit

        db.get_memories_by_type("SEMANTIC")
        assert col.find.called

    def test_invalid_type_raises(self):
        db, _ = _make_connected_db()
        with pytest.raises(ValueError):
            db.get_memories_by_type("INVALID_TYPE")

    def test_requires_connection(self):
        db = MemoryDatabase()
        with pytest.raises(RuntimeError, match="Not connected"):
            db.get_memories_by_type("SEMANTIC")


# ---------------------------------------------------------------------------
# MemoryDatabase.count_memories
# ---------------------------------------------------------------------------

class TestCountMemories:

    def test_count_all(self):
        db, col = _make_connected_db()
        col.count_documents.return_value = 42
        assert db.count_memories() == 42

    def test_count_by_type(self):
        db, col = _make_connected_db()
        col.count_documents.return_value = 10
        count = db.count_memories("SEMANTIC")
        assert count == 10
        call_args = col.count_documents.call_args[0][0]
        assert call_args == {"memory_type": "SEMANTIC"}


# ---------------------------------------------------------------------------
# MemoryDatabase.delete_memory
# ---------------------------------------------------------------------------

class TestDeleteMemory:

    def test_delete_existing(self):
        db, col = _make_connected_db()
        col.delete_one.return_value = MagicMock(deleted_count=1)
        with patch("database.mongodb.ObjectId") as mock_oid:
            mock_oid.return_value = "mock_oid_value"
            result = db.delete_memory("507f1f77bcf86cd799439011")
        assert result is True

    def test_delete_nonexistent(self):
        db, col = _make_connected_db()
        col.delete_one.return_value = MagicMock(deleted_count=0)
        with patch("database.mongodb.ObjectId") as mock_oid:
            mock_oid.return_value = "mock_oid_value"
            result = db.delete_memory("507f1f77bcf86cd799439011")
        assert result is False


# ---------------------------------------------------------------------------
# check_mongodb_connection
# ---------------------------------------------------------------------------

class TestCheckMongoDBConnection:

    @patch("database.mongodb.MongoClient")
    def test_connection_success(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.admin.command.return_value = {"ok": 1}
        result = check_mongodb_connection()
        assert result["connected"] is True
        assert result["error"] == ""

    @patch("database.mongodb.MongoClient")
    def test_connection_failure(self, mock_client_class):
        mock_client_class.side_effect = Exception("Connection refused")
        result = check_mongodb_connection()
        assert result["connected"] is False
        assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# MemoryDatabase.get_relevant_memories
# ---------------------------------------------------------------------------

class TestGetRelevantMemories:

    def test_empty_query_returns_empty_list(self):
        db, _ = _make_connected_db()
        result = db.get_relevant_memories("")
        assert result == []

    def test_none_query_returns_empty_list(self):
        db, _ = _make_connected_db()
        result = db.get_relevant_memories(None)
        assert result == []

    def test_requires_connection(self):
        db = MemoryDatabase()
        with pytest.raises(RuntimeError, match="Not connected"):
            db.get_relevant_memories("test query")
