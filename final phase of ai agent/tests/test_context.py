"""tests/test_context.py — Unit tests for context_analysis.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from context_analysis import analyze_context, CONTEXT_THRESHOLD


class TestAnalyzeContext:

    def test_empty_input_returns_zero(self):
        result = analyze_context("")
        assert result["context_score"] == 0.0
        assert result["is_relevant"] is False

    def test_whitespace_only_returns_zero(self):
        result = analyze_context("   ")
        assert result["context_score"] == 0.0
        assert result["is_relevant"] is False

    def test_noise_greeting_low_score(self):
        result = analyze_context("hello")
        assert result["context_score"] < CONTEXT_THRESHOLD
        assert result["is_relevant"] is False

    def test_noise_ok_low_score(self):
        result = analyze_context("ok")
        assert result["context_score"] < CONTEXT_THRESHOLD
        assert result["is_relevant"] is False

    def test_factual_statement_high_score(self):
        result = analyze_context("MongoDB is a document-oriented database used in many applications.")
        assert result["context_score"] >= CONTEXT_THRESHOLD
        assert result["is_relevant"] is True

    def test_procedural_statement_high_score(self):
        result = analyze_context("First analyze the memory, then classify it, then store it.")
        assert result["context_score"] >= CONTEXT_THRESHOLD
        assert result["is_relevant"] is True

    def test_episodic_statement_high_score(self):
        result = analyze_context("Yesterday our team decided to move the project to MongoDB.")
        assert result["context_score"] >= CONTEXT_THRESHOLD
        assert result["is_relevant"] is True

    def test_context_overlap_boosts_score(self):
        result_no_ctx = analyze_context("The database stores records efficiently.")
        result_with_ctx = analyze_context(
            "The database stores records efficiently.",
            recent_context="We are working with a database system."
        )
        # Score with context overlap should be >= score without
        assert result_with_ctx["context_score"] >= result_no_ctx["context_score"]

    def test_result_has_required_keys(self):
        result = analyze_context("Some test statement.")
        assert "context_score" in result
        assert "is_relevant" in result
        assert "reason" in result

    def test_score_within_bounds(self):
        for text in [
            "", "hi", "MongoDB is a database.", "Today I learned how APIs work.",
            "Step 1: connect. Step 2: query. Step 3: disconnect."
        ]:
            result = analyze_context(text)
            assert 0.0 <= result["context_score"] <= 1.0, f"Out of bounds for: {text!r}"

    def test_is_relevant_matches_threshold(self):
        result = analyze_context("MongoDB is a document-oriented database.")
        assert result["is_relevant"] == (result["context_score"] >= CONTEXT_THRESHOLD)
