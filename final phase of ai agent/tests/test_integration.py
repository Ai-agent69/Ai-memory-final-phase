"""End-to-end tests for the Ollama-first memory pipeline."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from longterm_memory import LONGTERM_THRESHOLD, decide_longterm
from main import process_input
from memory_evaluator import calculate_final_score, make_memory_decision


def _analysis(
    memory_type="SEMANTIC",
    long_term_beneficial=True,
    importance_score=0.9,
    persistence_score=0.9,
    usefulness_score=0.9,
    reason="Useful long-term memory.",
):
    return {
        "success": True,
        "memory_type": memory_type,
        "long_term_beneficial": long_term_beneficial,
        "importance_score": importance_score,
        "persistence_score": persistence_score,
        "usefulness_score": usefulness_score,
        "reason": reason,
        "model_used": "gemma4:12b",
        "error": "",
    }


class TestFinalScoreDecision:
    def test_formula_matches_required_weights(self):
        score = calculate_final_score(0.95, 0.99, 0.95)
        expected = round(0.95 * 0.35 + 0.99 * 0.35 + 0.95 * 0.30, 4)
        assert score == expected

    def test_threshold_default_is_point_seven(self):
        assert LONGTERM_THRESHOLD == 0.70

    def test_high_scores_pass(self):
        result = make_memory_decision(_analysis())
        assert result["is_longterm"] is True
        assert result["decision"] == "LONG_TERM"

    def test_low_scores_fail_even_if_beneficial_true(self):
        analysis = _analysis(
            memory_type="SEMANTIC",
            long_term_beneficial=True,
            importance_score=0.3,
            persistence_score=0.3,
            usefulness_score=0.3,
        )
        result = make_memory_decision(analysis)
        assert result["is_longterm"] is False
        assert result["decision"] == "TEMPORARY"

    def test_boolean_false_does_not_force_discard(self):
        analysis = _analysis(
            memory_type="SEMANTIC",
            long_term_beneficial=False,
            importance_score=0.95,
            persistence_score=0.95,
            usefulness_score=0.95,
        )
        result = make_memory_decision(analysis)
        assert result["is_longterm"] is True

    def test_legacy_decide_longterm_wrapper(self):
        result = decide_longterm(0.8, 0.8, 0.8)
        assert result["is_longterm"] is True
        assert result["final_score"] == result["combined_score"]


class TestProcessInputPipeline:
    def test_empty_input_not_analyzed_or_stored(self):
        result = process_input("")
        assert result["pipeline_complete"] is False
        assert result["analysis_success"] is False
        assert result["stored"] is False

    @patch("main.MemoryDatabase")
    @patch("main.analyze_memory")
    def test_semantic_long_term_stored(self, mock_analyze, mock_db_class):
        mock_analyze.return_value = _analysis(
            "SEMANTIC",
            True,
            0.95,
            0.99,
            0.95,
            "Stable identity information.",
        )
        mock_db = MagicMock()
        mock_db.insert_analyzed_memory.return_value = "507f1f77bcf86cd799439011"
        mock_db_class.return_value = mock_db

        result = process_input("My name is Sujay Das")

        assert result["pipeline_complete"] is True
        assert result["decision"] == "LONG_TERM"
        assert result["stored"] is True
        assert result["mongodb_status"] == "STORED"
        document = mock_db.insert_analyzed_memory.call_args[0][0]
        assert document["memory_type"] == "SEMANTIC"
        assert document["final_score"] == result["final_score"]

    @patch("main.MemoryDatabase")
    @patch("main.analyze_memory")
    def test_temporary_memory_not_inserted(self, mock_analyze, mock_db_class):
        mock_analyze.return_value = _analysis(
            "OTHER",
            False,
            0.1,
            0.1,
            0.1,
            "Temporary activity.",
        )

        result = process_input("I am currently eating food")

        assert result["decision"] == "TEMPORARY"
        assert result["stored"] is False
        assert result["mongodb_status"] == "NOT STORED"
        mock_db_class.assert_not_called()

    @patch("main.MemoryDatabase")
    @patch("main.analyze_memory")
    def test_mongodb_unavailable_does_not_crash(self, mock_analyze, mock_db_class):
        mock_analyze.return_value = _analysis("PROCEDURAL", True, 0.9, 0.9, 0.9)
        mock_db = MagicMock()
        mock_db.connect.side_effect = Exception("MongoDB not running")
        mock_db_class.return_value = mock_db

        result = process_input("First activate the environment and run main.py")

        assert result["decision"] == "LONG_TERM"
        assert result["stored"] is False
        assert result["mongodb_status"] == "UNAVAILABLE"
        assert "MongoDB not running" in result["storage_error"]

    @patch("main.analyze_memory")
    def test_ollama_error_stops_before_mongodb(self, mock_analyze):
        mock_analyze.return_value = {
            "success": False,
            "model_used": "gemma4:12b",
            "error": "Connection refused",
            "reason": "Could not communicate with Ollama.",
        }

        result = process_input("I like football")

        assert result["pipeline_complete"] is False
        assert result["analysis_success"] is False
        assert result["stored"] is False
        assert result["mongodb_status"] == "NOT_ATTEMPTED"

    @patch("main.MemoryDatabase")
    @patch("main.analyze_memory")
    def test_final_score_not_ollama_boolean_controls_storage(self, mock_analyze, mock_db_class):
        mock_analyze.return_value = _analysis(
            "SEMANTIC",
            True,
            0.2,
            0.2,
            0.2,
            "Beneficial signal is true but scores are low.",
        )

        result = process_input("Maybe remember this tiny thing")

        assert result["long_term_beneficial"] is True
        assert result["decision"] == "TEMPORARY"
        assert result["stored"] is False
        mock_db_class.assert_not_called()
