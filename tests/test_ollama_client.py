"""Unit tests for OllamaClient HTTP chat calls."""
from __future__ import annotations

import json
import socket
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from norn.domain.models import ModelConfig
from norn.runtime.ollama_client import OllamaClient, OllamaConnectionError


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def model_config():
    """Default ModelConfig for tests."""
    return ModelConfig()


@pytest.fixture
def client():
    """Fresh OllamaClient instance."""
    return OllamaClient()


def _fake_response_data(content: str = "test response",
                        prompt_eval: int = 10,
                        eval_count: int = 5,
                        total_duration: int = 500_000_000) -> bytes:
    """Build a fake Ollama JSON response as bytes."""
    return json.dumps({
        "model": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "done": True,
    }).encode("utf-8")


# ── URL / body construction ──────────────────────────────────────────────────

def test_url_uses_host_and_port(client, model_config):
    """POST is sent to http://{host}:{port}/api/chat."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    called_url = mock_open.call_args[0][0]
    assert isinstance(called_url, urllib.request.Request)
    assert called_url.full_url == "http://localhost:11434/api/chat"


def test_content_type_header_is_json(client, model_config):
    """Request includes Content-Type: application/json header."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.get_header("Content-type") == "application/json"


def test_custom_host_and_port():
    """Non-default host and port are reflected in the URL."""
    mc = ModelConfig(host="192.168.1.10", port=9999)
    client = OllamaClient()
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.full_url == "http://192.168.1.10:9999/api/chat"


# ── JSON body checks ─────────────────────────────────────────────────────────

def _extract_body(mock_open) -> dict:
    """Extract the deserialized JSON body from a mocked urlopen call."""
    called_req = mock_open.call_args[0][0]
    return json.loads(called_req.data.decode("utf-8"))


def test_body_includes_model_name(client, model_config):
    """Request body includes the model_name."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    body = _extract_body(mock_open)
    assert body["model"] == "llama3.1:8b"


def test_body_includes_user_message(client, model_config):
    """Request body includes the user prompt in messages[0].content."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello world")

    body = _extract_body(mock_open)
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello world"


def test_body_includes_keep_alive(client, model_config):
    """Request body includes keep_alive: '5m'."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    body = _extract_body(mock_open)
    assert body["keep_alive"] == "5m"


def test_body_stream_is_false(client, model_config):
    """Request body includes stream: false."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    body = _extract_body(mock_open)
    assert body["stream"] is False


def test_body_includes_options(client, model_config):
    """Request body includes options dict with generation parameters."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    body = _extract_body(mock_open)
    assert "options" in body
    opts = body["options"]
    assert opts["temperature"] == 0.0
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 2048


def test_seed_none_coerces_to_zero(client):
    """When seed is None, the body uses 0 (Ollama random seed)."""
    mc = ModelConfig(seed=None)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    body = _extract_body(mock_open)
    assert body["options"]["seed"] == 0


def test_seed_42_preserved(client):
    """When seed is 42, the body uses 42."""
    mc = ModelConfig(seed=42)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    body = _extract_body(mock_open)
    assert body["options"]["seed"] == 42


def test_num_predict_uses_max_tokens(client):
    """num_predict in options equals max_tokens from ModelConfig."""
    mc = ModelConfig(max_tokens=512)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    body = _extract_body(mock_open)
    assert body["options"]["num_predict"] == 512


# ── Response parsing ─────────────────────────────────────────────────────────

def test_response_text_extracted_from_message_content(client, model_config):
    """Response text comes from message.content in the JSON response."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(content="Hello, human!")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        response_text, *_ = result[:4]
    assert response_text == "Hello, human!"


def test_tokens_in_extracted_from_prompt_eval_count(client, model_config):
    """tokens_in comes from prompt_eval_count."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(prompt_eval=42)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        _, tokens_in, _, _ = result[:4]
    assert tokens_in == 42


def test_tokens_out_extracted_from_eval_count(client, model_config):
    """tokens_out comes from eval_count."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(eval_count=7)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        _, _, tokens_out, _ = result[:4]
    assert tokens_out == 7


def test_latency_computed_from_total_duration(client, model_config):
    """latency_ms = total_duration_ns / 1_000_000."""
    # 2 seconds in nanoseconds
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(total_duration=2_000_000_000)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        _, _, _, latency_ms = result[:4]
    assert latency_ms == 2000.0


def test_latency_is_float(client, model_config):
    """latency_ms is a float, not an int."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(total_duration=500_000_000)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        _, _, _, latency_ms = result[:4]
    assert isinstance(latency_ms, float)
    assert latency_ms == 500.0


# ── Error handling ───────────────────────────────────────────────────────────

def test_connection_failure_raises_runtime_error(client, model_config):
    """A URLError with non-connection-refused reason raises RuntimeError."""
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("Unknown host")):
        with pytest.raises(RuntimeError, match="Failed to connect"):
            client.chat(model_config, "hello")


def test_connection_refused_raises_ollama_connection_error(client, model_config):
    """A URLError with ConnectionRefusedError reason raises OllamaConnectionError."""
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError(ConnectionRefusedError("refused"))):
        with pytest.raises(OllamaConnectionError, match="Is Ollama running?"):
            client.chat(model_config, "hello")


def test_socket_timeout_raises_timeout_error(client, model_config):
    """A socket timeout raises TimeoutError."""
    with patch("urllib.request.urlopen",
               side_effect=socket.timeout("timed out")):
        with pytest.raises(TimeoutError, match="timed out"):
            client.chat(model_config, "hello")


def test_os_error_raises_runtime_error(client, model_config):
    """An OSError without errno 111 raises RuntimeError."""
    with patch("urllib.request.urlopen",
               side_effect=OSError("No route to host")):
        with pytest.raises(RuntimeError, match="Failed to connect"):
            client.chat(model_config, "hello")


def test_invalid_json_raises_runtime_error(client, model_config):
    """Non-JSON response raises RuntimeError."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"not json at all!!!"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Invalid response"):
            client.chat(model_config, "hello")


def test_http_404_raises_runtime_error_with_model_name(client, model_config):
    """HTTP 404 raises RuntimeError with model name and 'not found'."""
    mock_http_error = urllib.error.HTTPError(
        "http://localhost:11434/api/chat", 404, "Not Found", {}, None
    )
    # Patch read to return empty bytes for error body
    mock_http_error.read = MagicMock(return_value=b"{}")
    with patch("urllib.request.urlopen",
               side_effect=mock_http_error):
        with pytest.raises(RuntimeError, match="not found on Ollama server"):
            client.chat(model_config, "hello")


def test_http_500_raises_runtime_error(client, model_config):
    """HTTP 500 raises RuntimeError with status code."""
    mock_http_error = urllib.error.HTTPError(
        "http://localhost:11434/api/chat", 500, "Internal Server Error", {}, None
    )
    with patch("urllib.request.urlopen",
               side_effect=mock_http_error):
        with pytest.raises(RuntimeError, match="server error.*HTTP 500"):
            client.chat(model_config, "hello")


def test_error_messages_include_host_and_port(client):
    """All error messages include host:port information."""
    mc = ModelConfig(host="192.168.1.10", port=9999)
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError(ConnectionRefusedError("refused"))):
        with pytest.raises(OllamaConnectionError, match="192.168.1.10:9999"):
            client.chat(mc, "hello")


# ── Timeout passthrough ──────────────────────────────────────────────────────

def test_timeout_passed_to_urlopen(client, model_config):
    """The timeout kwarg is passed through to urlopen."""
    mc = ModelConfig(timeout=30.0)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    assert mock_open.call_args[1]["timeout"] == 30.0


def test_default_timeout_is_60_seconds(client, model_config):
    """Default timeout is 60.0 when not overridden."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    assert mock_open.call_args[1]["timeout"] == 60.0


# ── list_models classmethod ──────────────────────────────────────────────────

def test_list_models_returns_model_names():
    """list_models returns list of model name strings."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps({
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "mistral:7b"},
            {"name": "gemma:2b"},
        ]
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = OllamaClient.list_models("localhost", 11434)
    assert models == ["llama3.1:8b", "mistral:7b", "gemma:2b"]


def test_list_models_empty_when_no_models():
    """list_models returns empty list when server has no models."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps({"models": []}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = OllamaClient.list_models("localhost", 11434)
    assert models == []


def test_list_models_uses_api_tags_endpoint():
    """list_models calls GET /api/tags on the configured host:port."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps({"models": []}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        OllamaClient.list_models("192.168.1.10", 9999, timeout=5.0)

    called_req = mock_open.call_args[0][0]
    assert called_req.full_url == "http://192.168.1.10:9999/api/tags"
    assert mock_open.call_args[1]["timeout"] == 5.0


def test_list_models_default_timeout_is_5_seconds():
    """list_models uses 5-second default timeout."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps({"models": []}).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        OllamaClient.list_models("localhost", 11434)

    assert mock_open.call_args[1]["timeout"] == 5.0


def test_ollama_connection_error_is_connection_error_subclass():
    """OllamaConnectionError inherits from ConnectionError."""
    assert issubclass(OllamaConnectionError, ConnectionError)
