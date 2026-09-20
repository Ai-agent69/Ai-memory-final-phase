"""tests/test_memory_type.py — Unit tests for memory_type classification (mocked Ollama)"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai.ollama_client import (
    classify_memory,
    check_ollama_connection,
    _extract_json,
    _validate_response,
    VALID_MEMORY_TYPES,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:

    def test_clean_json(self):
        text = '{"memory_type": "SEMANTIC", "confidence": 0.9, "reason": "fact"}'
        result = _extract_json(text)
        assert result is not None
        assert result["memory_type"] == "SEMANTIC"

    def test_json_with_markdown_fences(self):
        text = '```json\n{"memory_type": "EPISODIC", "confidence": 0.8, "reason": "event"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["memory_type"] == "EPISODIC"

    def test_json_with_preamble(self):
        text = 'Sure! Here is the JSON:\n{"memory_type": "PROCEDURAL", "confidence": 0.85, "reason": "how-to"}'
        result = _extract_json(text)
        assert result is not None
        assert result["memory_type"] == "PROCEDURAL"

    def test_invalid_json_returns_none(self):
        result = _extract_json("This is not JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _extract_json("")
        assert result is None

    def test_none_returns_none(self):
        result = _extract_json(None)
        assert result is None


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------

class TestValidateResponse:

    def _valid(self):
        return {"memory_type": "SEMANTIC", "confidence": 0.9, "reason": "test reason"}

    def test_valid_semantic(self):
        is_valid, err = _validate_response(self._valid())
        assert is_valid is True
        assert err == ""

    def test_valid_episodic(self):
        d = {"memory_type": "EPISODIC", "confidence": 0.75, "reason": "event"}
        is_valid, _ = _validate_response(d)
        assert is_valid is True

    def test_valid_procedural(self):
        d = {"memory_type": "PROCEDURAL", "confidence": 0.60, "reason": "procedure"}
        is_valid, _ = _validate_response(d)
        assert is_valid is True

    def test_invalid_memory_type(self):
        d = {"memory_type": "UNKNOWN", "confidence": 0.9, "reason": "test"}
        is_valid, err = _validate_response(d)
        assert is_valid is False
        assert "UNKNOWN" in err

    def test_missing_memory_type(self):
        d = {"confidence": 0.9, "reason": "test"}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_confidence_out_of_range_high(self):
        d = {"memory_type": "SEMANTIC", "confidence": 1.5, "reason": "test"}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_confidence_out_of_range_low(self):
        d = {"memory_type": "SEMANTIC", "confidence": -0.1, "reason": "test"}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_missing_confidence(self):
        d = {"memory_type": "SEMANTIC", "reason": "test"}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_empty_reason(self):
        d = {"memory_type": "SEMANTIC", "confidence": 0.8, "reason": ""}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_missing_reason(self):
        d = {"memory_type": "SEMANTIC", "confidence": 0.8}
        is_valid, _ = _validate_response(d)
        assert is_valid is False

    def test_not_a_dict(self):
        is_valid, _ = _validate_response("string")
        assert is_valid is False


# ---------------------------------------------------------------------------
# classify_memory — mocked Ollama
# ---------------------------------------------------------------------------

class TestClassifyMemory:

    def _mock_response(self, memory_type="SEMANTIC", confidence=0.9, reason="test"):
        import json
        content = json.dumps({
            "memory_type": memory_type,
            "confidence": confidence,
            "reason": reason,
        })
        mock = MagicMock()
        mock.__getitem__ = lambda s, k: {"message": {"content": content}}[k]
        return mock

    @patch("ai.ollama_client.ollama.chat")
    def test_semantic_classification(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "SEMANTIC", "confidence": 0.92, "reason": "factual"}'
            }
        }
        result = classify_memory("MongoDB is a document-oriented database.")
        assert result["success"] is True
        assert result["memory_type"] == "SEMANTIC"
        assert result["confidence"] == 0.92
        assert result["error"] == ""

    @patch("ai.ollama_client.ollama.chat")
    def test_episodic_classification(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "EPISODIC", "confidence": 0.88, "reason": "event"}'
            }
        }
        result = classify_memory("Yesterday the team decided to use MongoDB.")
        assert result["success"] is True
        assert result["memory_type"] == "EPISODIC"

    @patch("ai.ollama_client.ollama.chat")
    def test_procedural_classification(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "PROCEDURAL", "confidence": 0.85, "reason": "how-to"}'
            }
        }
        result = classify_memory("First connect, then query, then close.")
        assert result["success"] is True
        assert result["memory_type"] == "PROCEDURAL"

    @patch("ai.ollama_client.ollama.chat")
    def test_invalid_json_handled_safely(self, mock_chat):
        mock_chat.return_value = {
            "message": {"content": "Sorry, I cannot classify this."}
        }
        result = classify_memory("Test statement.")
        assert result["success"] is False
        assert result["memory_type"] is None
        assert result["error"] != ""

    @patch("ai.ollama_client.ollama.chat")
    def test_invalid_memory_type_handled(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "UNKNOWN", "confidence": 0.9, "reason": "bad"}'
            }
        }
        result = classify_memory("Test statement.")
        assert result["success"] is False

    @patch("ai.ollama_client.ollama.chat")
    def test_confidence_out_of_range_handled(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "SEMANTIC", "confidence": 2.5, "reason": "test"}'
            }
        }
        result = classify_memory("Test statement.")
        assert result["success"] is False

    @patch("ai.ollama_client.ollama.chat")
    def test_connection_error_handled(self, mock_chat):
        mock_chat.side_effect = Exception("Connection refused")
        result = classify_memory("Test statement.")
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    def test_empty_statement(self):
        result = classify_memory("")
        assert result["success"] is False
        assert result["memory_type"] is None

    @patch("ai.ollama_client.ollama.chat")
    def test_with_relevant_memories_context(self, mock_chat):
        """Test that relevant memories are included in prompt without crashing."""
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "PROCEDURAL", "confidence": 0.80, "reason": "same method"}'
            }
        }
        relevant = [
            {"memory": "How to connect Flask to MongoDB.", "memory_type": "PROCEDURAL"}
        ]
        result = classify_memory("I want to use the same method.", relevant_memories=relevant)
        assert result["success"] is True
        # Verify the prompt included context
        call_args = mock_chat.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "Flask" in prompt

    @patch("ai.ollama_client.ollama.chat")
    def test_memory_type_uppercase_normalisation(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"memory_type": "semantic", "confidence": 0.85, "reason": "fact"}'
            }
        }
        result = classify_memory("Python is a language.")
        # The validator calls .upper() on memory_type, so "semantic" → "SEMANTIC" is valid.
        # This is correct lenient behavior — LLMs sometimes return lowercase.
        assert result["success"] is True
        assert result["memory_type"] == "SEMANTIC"  # normalised to uppercase in return value


# ---------------------------------------------------------------------------
# check_ollama_connection — mocked
# ---------------------------------------------------------------------------

class TestCheckOllamaConnection:

    @patch("ai.ollama_client.ollama.list")
    def test_connected_model_found(self, mock_list):
        mock_list.return_value = {
            "models": [{"model": "gemma4:12b"}, {"model": "gemma4:latest"}]
        }
        import ai.ollama_client as oc
        original = oc.OLLAMA_MODEL
        oc.OLLAMA_MODEL = "gemma4:12b"
        result = check_ollama_connection()
        oc.OLLAMA_MODEL = original
        assert result["connected"] is True
        assert result["model_found"] is True

    @patch("ai.ollama_client.ollama.list")
    def test_connected_model_not_found(self, mock_list):
        mock_list.return_value = {
            "models": [{"model": "gemma4:latest"}]
        }
        import ai.ollama_client as oc
        original = oc.OLLAMA_MODEL
        oc.OLLAMA_MODEL = "nonexistent:model"
        result = check_ollama_connection()
        oc.OLLAMA_MODEL = original
        assert result["connected"] is True
        assert result["model_found"] is False

    @patch("ai.ollama_client.ollama.list")
    def test_connection_failure(self, mock_list):
        mock_list.side_effect = Exception("Connection refused")
        result = check_ollama_connection()
        assert result["connected"] is False
        assert "Connection refused" in result["error"]
