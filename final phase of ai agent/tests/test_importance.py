"""tests/test_importance.py — Unit tests for importance_analysis.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from importance_analysis import analyze_importance, IMPORTANCE_THRESHOLD


class TestAnalyzeImportance:

    def test_empty_input_returns_zero(self):
        result = analyze_importance("")
        assert result["importance_score"] == 0.0
        assert result["is_important"] is False

    def test_whitespace_only_returns_zero(self):
        result = analyze_importance("   ")
        assert result["importance_score"] == 0.0
        assert result["is_important"] is False

    def test_factual_definition_high_importance(self):
        result = analyze_importance("MongoDB is a document-oriented NoSQL database.")
        assert result["importance_score"] >= IMPORTANCE_THRESHOLD
        assert result["is_important"] is True

    def test_procedural_steps_high_importance(self):
        result = analyze_importance(
            "First connect to the database, then run the query, finally close the connection."
        )
        assert result["importance_score"] >= IMPORTANCE_THRESHOLD
        assert result["is_important"] is True

    def test_uncertainty_language_lowers_score(self):
        result_clear = analyze_importance("Python is a programming language.")
        result_hedge = analyze_importance("I think Python might be a programming language maybe.")
        assert result_clear["importance_score"] > result_hedge["importance_score"]

    def test_named_entity_boosts_score(self):
        result_generic = analyze_importance("this is a database tool")
        result_named = analyze_importance("MongoDB is a database tool.")
        assert result_named["importance_score"] >= result_generic["importance_score"]

    def test_result_has_required_keys(self):
        result = analyze_importance("Some test statement.")
        assert "importance_score" in result
        assert "is_important" in result
        assert "reason" in result

    def test_score_within_bounds(self):
        for text in [
            "", "hi", "MongoDB is defined as a NoSQL database.", "maybe tomorrow",
            "I think this could be important possibly", "API is the interface."
        ]:
            result = analyze_importance(text)
            assert 0.0 <= result["importance_score"] <= 1.0, f"Out of bounds for: {text!r}"

    def test_is_important_matches_threshold(self):
        result = analyze_importance("Python is a high-level programming language.")
        assert result["is_important"] == (result["importance_score"] >= IMPORTANCE_THRESHOLD)

    def test_technical_terms_boost_score(self):
        result = analyze_importance(
            "The REST API uses JSON schema with a defined protocol."
        )
        assert result["importance_score"] >= IMPORTANCE_THRESHOLD
