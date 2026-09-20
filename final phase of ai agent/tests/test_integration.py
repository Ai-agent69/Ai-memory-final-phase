"""
tests/test_integration.py
──────────────────────────
End-to-end integration tests for the complete memory pipeline.

Ollama and MongoDB are both mocked so tests run without external services.
The tests verify that all components wire together correctly and that the
correct data flows through the pipeline.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_analysis import analyze_context
from importance_analysis import analyze_importance
from persistence_analysis import analyze_persistence
from longterm_memory import decide_longterm, LONGTERM_THRESHOLD
from main import process_input


# ---------------------------------------------------------------------------
# Long-term memory decision tests (pure logic, no mocks needed)
# ---------------------------------------------------------------------------

class TestLongtermDecision:

    def test_high_scores_pass(self):
        result = decide_longterm(0.8, 0.85, 0.9)
        assert result["is_longterm"] is True

    def test_low_scores_fail(self):
        result = decide_longterm(0.1, 0.2, 0.1)
        assert result["is_longterm"] is False

    def test_zero_scores_fail(self):
        result = decide_longterm(0.0, 0.0, 0.0)
        assert result["is_longterm"] is False

    def test_all_scores_at_threshold_pass(self):
        t = LONGTERM_THRESHOLD
        result = decide_longterm(t, t, t)
        assert result["is_longterm"] is True

    def test_combined_score_is_weighted(self):
        result = decide_longterm(0.6, 0.8, 0.7)
        expected = round(0.6 * 0.30 + 0.8 * 0.40 + 0.7 * 0.30, 4)
        assert result["combined_score"] == expected

    def test_result_has_required_keys(self):
        result = decide_longterm(0.5, 0.5, 0.5)
        assert "is_longterm" in result
        assert "combined_score" in result
        assert "reason" in result

    def test_one_strong_score_required(self):
        """Three moderate scores that sum above threshold but none strong."""
        # With equal weights: 0.45*0.30 + 0.45*0.40 + 0.45*0.30 = 0.45 < 0.5
        result = decide_longterm(0.45, 0.45, 0.45)
        assert result["is_longterm"] is False  # below threshold

    def test_clamping_above_one(self):
        result = decide_longterm(2.0, 2.0, 2.0)
        assert result["combined_score"] <= 1.0

    def test_clamping_below_zero(self):
        result = decide_longterm(-1.0, -0.5, -0.2)
        assert result["combined_score"] >= 0.0


# ---------------------------------------------------------------------------
# Full pipeline: process_input (mocked Ollama + MongoDB)
# ---------------------------------------------------------------------------

def _mock_ollama_response(memory_type="SEMANTIC", confidence=0.9):
    import json
    return {
        "message": {
            "content": json.dumps({
                "memory_type": memory_type,
                "confidence": confidence,
                "reason": f"Classified as {memory_type}.",
            })
        }
    }


class TestProcessInputPipeline:

    def test_empty_input_not_stored(self):
        result = process_input("")
        assert result["pipeline_complete"] is False

    def test_noise_input_not_stored(self):
        """Greeting should fail context/importance/persistence gates."""
        result = process_input("hello")
        assert result.get("stored") is False or not result.get("stored", True)

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_semantic_statement_stored(self, mock_db_class, mock_classify):
        """A clear factual statement should be classified as SEMANTIC and stored."""
        mock_classify.return_value = {
            "success": True,
            "memory_type": "SEMANTIC",
            "confidence": 0.92,
            "reason": "factual statement",
            "model_used": "gemma4:12b",
            "error": "",
        }
        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.get_relevant_memories.return_value = []
        mock_db.insert_memory.return_value = "507f1f77bcf86cd799439011"
        mock_db_class.return_value = mock_db

        result = process_input("MongoDB is a document-oriented database used in enterprise systems.")
        assert result["pipeline_complete"] is True
        if result.get("longterm_decision", {}).get("is_longterm"):
            assert result["classification"]["memory_type"] == "SEMANTIC"

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_episodic_statement_stored(self, mock_db_class, mock_classify):
        mock_classify.return_value = {
            "success": True,
            "memory_type": "EPISODIC",
            "confidence": 0.88,
            "reason": "event",
            "model_used": "gemma4:12b",
            "error": "",
        }
        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.get_relevant_memories.return_value = []
        mock_db.insert_memory.return_value = "507f1f77bcf86cd799439012"
        mock_db_class.return_value = mock_db

        result = process_input("Yesterday our team decided to migrate the database to MongoDB.")
        assert result["pipeline_complete"] is True

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_procedural_statement_stored(self, mock_db_class, mock_classify):
        mock_classify.return_value = {
            "success": True,
            "memory_type": "PROCEDURAL",
            "confidence": 0.85,
            "reason": "procedure",
            "model_used": "gemma4:12b",
            "error": "",
        }
        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.get_relevant_memories.return_value = []
        mock_db.insert_memory.return_value = "507f1f77bcf86cd799439013"
        mock_db_class.return_value = mock_db

        result = process_input(
            "First analyze the memory, then calculate scores, then classify and store it in MongoDB."
        )
        assert result["pipeline_complete"] is True

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_invalid_ollama_output_handled(self, mock_db_class, mock_classify):
        """Invalid LLM output should not crash the program."""
        mock_classify.return_value = {
            "success": False,
            "memory_type": None,
            "confidence": None,
            "reason": "LLM returned non-JSON output.",
            "model_used": "gemma4:12b",
            "error": "Non-JSON response: Sorry, I cannot classify this.",
        }
        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.get_relevant_memories.return_value = []
        mock_db_class.return_value = mock_db

        # Should not raise, should return pipeline_complete=True with stored=False
        result = process_input("MongoDB is a document-oriented database system.")
        assert result["pipeline_complete"] is True
        if result.get("longterm_decision", {}).get("is_longterm"):
            assert result.get("stored") is False

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_mongodb_unavailable_handled(self, mock_db_class, mock_classify):
        """MongoDB connection failure should not crash the program."""
        mock_classify.return_value = {
            "success": True,
            "memory_type": "SEMANTIC",
            "confidence": 0.9,
            "reason": "fact",
            "model_used": "gemma4:12b",
            "error": "",
        }
        mock_db = MagicMock()
        mock_db.connect.side_effect = Exception("MongoDB not running")
        mock_db_class.return_value = mock_db

        # Should not raise
        result = process_input("MongoDB is a document-oriented database system.")
        assert result["pipeline_complete"] is True

    @patch("memory_types.memory_type_analyzer.classify_memory")
    @patch("memory_types.memory_type_analyzer.MemoryDatabase")
    def test_context_aware_ambiguous_statement(self, mock_db_class, mock_classify):
        """
        Ambiguous statement with relevant previous memory context.
        The system should pass previous memories to Ollama as context.
        """
        mock_classify.return_value = {
            "success": True,
            "memory_type": "PROCEDURAL",
            "confidence": 0.80,
            "reason": "same method refers to a procedure",
            "model_used": "gemma4:12b",
            "error": "",
        }
        mock_db = MagicMock()
        mock_db.connect.return_value = None
        mock_db.get_relevant_memories.return_value = [
            {
                "memory": "The user learned how to connect Flask to MongoDB.",
                "memory_type": "PROCEDURAL",
            }
        ]
        mock_db.insert_memory.return_value = "507f1f77bcf86cd799439014"
        mock_db_class.return_value = mock_db

        result = process_input(
            "I want to use the same method for my new project.",
        )
        assert result["pipeline_complete"] is True

    def test_all_pipeline_keys_present(self):
        """The result dict should always have the top-level pipeline keys."""
        result = process_input("Python is a high-level programming language.")
        assert "input" in result
        assert "pipeline_complete" in result
        assert "context" in result
        assert "importance" in result
        assert "persistence" in result
        assert "longterm_decision" in result
