"""Tests for Ollama structured memory analysis."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.ollama_client import (
    VALID_MEMORY_TYPES,
    _build_messages,
    _extract_json,
    _validate_response,
    analyze_memory,
    check_ollama_connection,
    validate_memory_analysis,
)


def _analysis_payload(
    memory_type="SEMANTIC",
    long_term_beneficial=True,
    importance_score=0.9,
    persistence_score=0.9,
    usefulness_score=0.9,
    reason="Stable and useful memory.",
):
    return {
        "memory_type": memory_type,
        "long_term_beneficial": long_term_beneficial,
        "importance_score": importance_score,
        "persistence_score": persistence_score,
        "usefulness_score": usefulness_score,
        "reason": reason,
    }


def _mock_client_for_payload(payload):
    client = MagicMock()
    client.chat.return_value = {"message": {"content": json.dumps(payload)}}
    return client


class TestExtractJson:
    def test_clean_json(self):
        payload = _analysis_payload()
        result = _extract_json(json.dumps(payload))
        assert result == payload

    def test_json_with_markdown_fences(self):
        payload = _analysis_payload(memory_type="EPISODIC")
        result = _extract_json(f"```json\n{json.dumps(payload)}\n```")
        assert result["memory_type"] == "EPISODIC"

    def test_json_with_preamble(self):
        payload = _analysis_payload(memory_type="PROCEDURAL")
        result = _extract_json(f"Here is the JSON:\n{json.dumps(payload)}")
        assert result["memory_type"] == "PROCEDURAL"

    def test_invalid_json_returns_none(self):
        assert _extract_json("This is not JSON.") is None

    def test_empty_or_none_returns_none(self):
        assert _extract_json("") is None
        assert _extract_json(None) is None


class TestValidateMemoryAnalysis:
    def test_valid_payload(self):
        validated, error = validate_memory_analysis(_analysis_payload())
        assert error == ""
        assert validated["memory_type"] == "SEMANTIC"

    def test_valid_other_type(self):
        validated, error = validate_memory_analysis(
            _analysis_payload(memory_type="other", long_term_beneficial=False)
        )
        assert error == ""
        assert validated["memory_type"] == "OTHER"
        assert "OTHER" in VALID_MEMORY_TYPES

    @pytest.mark.parametrize(
        "bad_payload, expected_error",
        [
            ({"memory_type": "SEMANTIC"}, "Missing required"),
            (_analysis_payload(memory_type="UNKNOWN"), "memory_type"),
            (_analysis_payload(long_term_beneficial="true"), "boolean"),
            (_analysis_payload(importance_score=2.0), "importance_score"),
            (_analysis_payload(persistence_score=-0.1), "persistence_score"),
            (_analysis_payload(usefulness_score=True), "usefulness_score"),
            (_analysis_payload(reason=""), "reason"),
        ],
    )
    def test_invalid_payloads(self, bad_payload, expected_error):
        validated, error = validate_memory_analysis(bad_payload)
        assert validated is None
        assert expected_error in error

    def test_legacy_validate_response_tuple(self):
        is_valid, error = _validate_response(_analysis_payload())
        assert is_valid is True
        assert error == ""


class TestAnalyzeMemory:
    @pytest.mark.parametrize(
        "statement,payload,expected_type,expected_beneficial",
        [
            (
                "My name is Sujay Das",
                _analysis_payload("SEMANTIC", True, 0.95, 0.99, 0.95),
                "SEMANTIC",
                True,
            ),
            (
                "I am a 4th year AI and Data Science student",
                _analysis_payload("SEMANTIC", True, 0.9, 0.9, 0.9),
                "SEMANTIC",
                True,
            ),
            (
                "I like football",
                _analysis_payload("SEMANTIC", True, 0.72, 0.8, 0.76),
                "SEMANTIC",
                True,
            ),
            (
                "Yesterday I attended a college hackathon",
                _analysis_payload("EPISODIC", False, 0.45, 0.35, 0.35),
                "EPISODIC",
                False,
            ),
            (
                "I am currently eating food",
                _analysis_payload("OTHER", False, 0.15, 0.1, 0.1),
                "OTHER",
                False,
            ),
            (
                "First activate the virtual environment and then run main.py",
                _analysis_payload("PROCEDURAL", True, 0.85, 0.9, 0.88),
                "PROCEDURAL",
                True,
            ),
            (
                "I am working on a Hybrid Agentic Memory System using Python and MongoDB",
                _analysis_payload("SEMANTIC", True, 0.92, 0.86, 0.94),
                "SEMANTIC",
                True,
            ),
            (
                "What is Python?",
                _analysis_payload("OTHER", False, 0.2, 0.2, 0.15),
                "OTHER",
                False,
            ),
        ],
    )
    @patch("ai.ollama_client._get_client")
    def test_required_examples_with_mocked_ollama(
        self,
        mock_get_client,
        statement,
        payload,
        expected_type,
        expected_beneficial,
    ):
        mock_get_client.return_value = _mock_client_for_payload(payload)
        result = analyze_memory(statement)
        assert result["success"] is True
        assert result["memory_type"] == expected_type
        assert result["long_term_beneficial"] is expected_beneficial

    @patch("ai.ollama_client._get_client")
    def test_invalid_json_handled_safely(self, mock_get_client):
        client = MagicMock()
        client.chat.return_value = {"message": {"content": "not json"}}
        mock_get_client.return_value = client

        result = analyze_memory("Test statement.")
        assert result["success"] is False
        assert "Malformed" in result["error"]

    @patch("ai.ollama_client._get_client")
    def test_invalid_score_handled_safely(self, mock_get_client):
        mock_get_client.return_value = _mock_client_for_payload(
            _analysis_payload(importance_score=1.5)
        )
        result = analyze_memory("Test statement.")
        assert result["success"] is False
        assert "importance_score" in result["error"]

    @patch("ai.ollama_client._get_client")
    def test_connection_error_handled(self, mock_get_client):
        client = MagicMock()
        client.chat.side_effect = Exception("Connection refused")
        mock_get_client.return_value = client

        result = analyze_memory("Test statement.")
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    def test_empty_statement(self):
        result = analyze_memory("")
        assert result["success"] is False
        assert "Empty input" in result["error"]

    def test_prompt_requires_semantic_understanding(self):
        messages = _build_messages("I like football")
        prompt = messages[0]["content"]
        assert "Do not use keyword matching" in prompt
        assert "Return JSON only" in prompt


class TestCheckOllamaConnection:
    @patch("ai.ollama_client._get_client")
    def test_connected_model_found(self, mock_get_client):
        client = MagicMock()
        client.list.return_value = {
            "models": [{"model": "gemma4:12b"}, {"model": "gemma4:latest"}]
        }
        mock_get_client.return_value = client

        import ai.ollama_client as oc

        original = oc.OLLAMA_MODEL
        oc.OLLAMA_MODEL = "gemma4:12b"
        result = check_ollama_connection()
        oc.OLLAMA_MODEL = original

        assert result["connected"] is True
        assert result["model_found"] is True

    @patch("ai.ollama_client._get_client")
    def test_connection_failure(self, mock_get_client):
        client = MagicMock()
        client.list.side_effect = Exception("Connection refused")
        mock_get_client.return_value = client

        result = check_ollama_connection()
        assert result["connected"] is False
        assert "Connection refused" in result["error"]
