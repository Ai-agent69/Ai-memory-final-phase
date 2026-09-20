"""tests/test_persistence.py — Unit tests for persistence_analysis.py"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from persistence_analysis import analyze_persistence, PERSISTENCE_THRESHOLD


class TestAnalyzePersistence:

    def test_empty_input_returns_zero(self):
        result = analyze_persistence("")
        assert result["persistence_score"] == 0.0
        assert result["is_persistent"] is False

    def test_whitespace_only_returns_zero(self):
        result = analyze_persistence("   ")
        assert result["persistence_score"] == 0.0
        assert result["is_persistent"] is False

    def test_timeless_fact_high_persistence(self):
        result = analyze_persistence("MongoDB is a document-oriented database.")
        assert result["persistence_score"] >= PERSISTENCE_THRESHOLD
        assert result["is_persistent"] is True

    def test_procedural_knowledge_high_persistence(self):
        result = analyze_persistence(
            "How to connect to MongoDB: first install pymongo, then create a MongoClient."
        )
        assert result["persistence_score"] >= PERSISTENCE_THRESHOLD

    def test_time_sensitive_low_persistence(self):
        result = analyze_persistence("The meeting is tomorrow at 3pm.")
        assert result["persistence_score"] < PERSISTENCE_THRESHOLD

    def test_ephemeral_signal_reduces_score(self):
        result_stable = analyze_persistence("Python is a programming language.")
        result_ephemeral = analyze_persistence("Right now today Python is being used.")
        assert result_stable["persistence_score"] > result_ephemeral["persistence_score"]

    def test_personal_opinion_reduces_score(self):
        result_fact = analyze_persistence("MongoDB stores data as documents.")
        result_opinion = analyze_persistence("I think MongoDB is pretty good.")
        assert result_fact["persistence_score"] > result_opinion["persistence_score"]

    def test_result_has_required_keys(self):
        result = analyze_persistence("Some statement.")
        assert "persistence_score" in result
        assert "is_persistent" in result
        assert "reason" in result

    def test_score_within_bounds(self):
        for text in [
            "", "today", "MongoDB is defined as a NoSQL database.",
            "I feel this might change tomorrow", "The algorithm processes data."
        ]:
            result = analyze_persistence(text)
            assert 0.0 <= result["persistence_score"] <= 1.0, f"Out of bounds for: {text!r}"

    def test_is_persistent_matches_threshold(self):
        result = analyze_persistence("REST APIs use HTTP protocol.")
        assert result["is_persistent"] == (result["persistence_score"] >= PERSISTENCE_THRESHOLD)

    def test_date_pattern_reduces_score(self):
        result_no_date = analyze_persistence("The team uses Python for backend development.")
        result_with_date = analyze_persistence("On 12/05/2024 the team decided to use Python.")
        # Date-bound statement should be less persistent
        assert result_no_date["persistence_score"] >= result_with_date["persistence_score"]
