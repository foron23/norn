"""Unit tests for OpenAICompatibleClient HTTP chat calls."""
from __future__ import annotations

import json
import socket
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from norn.domain.models import ModelConfig
from norn.runtime.openai_client import OpenAICompatibleClient


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def model_config():
    """Default ModelConfig with openai provider."""
    return ModelConfig(provider="openai")


@pytest.fixture
def client():
    """Fresh OpenAICompatibleClient instance."""
    return OpenAICompatibleClient()


# ── Utilities ────────────────────────────────────────────────────────────────

def _fake_response_data(content: str = "test response",
                        prompt_tokens: int = 10,
                        completion_tokens: int = 5) -> bytes:
    """Build a fake OpenAI Chat Completions JSON response as bytes."""
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }).encode("utf-8")


def _extract_body(mock_open) -> dict:
    """Extract the deserialized JSON body from a mocked urlopen call."""
    called_req = mock_open.call_args[0][0]
    return json.loads(called_req.data.decode("utf-8"))


# ── URL / header construction ────────────────────────────────────────────────

def test_url_uses_base_url(client, model_config):
    """POST is sent to {base_url}/chat/completions."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    called_req = mock_open.call_args[0][0]
    assert isinstance(called_req, urllib.request.Request)
    assert called_req.full_url == "https://api.openai.com/v1/chat/completions"


def test_custom_base_url():
    """Non-default base_url is reflected in the URL."""
    mc = ModelConfig(provider="openai", base_url="http://localhost:8080/v1")
    client = OpenAICompatibleClient()
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.full_url == "http://localhost:8080/v1/chat/completions"


def test_trailing_slash_stripped():
    """base_url with trailing slash produces single /chat/completions."""
    mc = ModelConfig(provider="openai", base_url="http://localhost:8080/v1/")
    client = OpenAICompatibleClient()
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.full_url == "http://localhost:8080/v1/chat/completions"
    assert "//chat" not in called_req.full_url


def test_content_type_header_is_json(client, model_config):
    """Request includes Content-Type: application/json header."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.get_header("Content-type") == "application/json"


def test_auth_header_when_api_key_set():
    """Authorization: Bearer header sent when api_key is set."""
    mc = ModelConfig(provider="openai", api_key="sk-test-1234")
    client = OpenAICompatibleClient()
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.get_header("Authorization") == "Bearer sk-test-1234"


def test_no_auth_header_when_api_key_none(client, model_config):
    """No Authorization header when api_key is None."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    called_req = mock_open.call_args[0][0]
    assert called_req.get_header("Authorization") is None


# ── JSON body checks ─────────────────────────────────────────────────────────

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


def test_body_includes_temperature_top_p_max_tokens(client, model_config):
    """Request body includes temperature, top_p, and max_tokens as flat keys."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(model_config, "hello")

    body = _extract_body(mock_open)
    assert body["temperature"] == 0.0
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 2048


def test_seed_none_omitted(client):
    """When seed is None, seed key is not in request body."""
    mc = ModelConfig(provider="openai", seed=None)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    body = _extract_body(mock_open)
    assert "seed" not in body


def test_seed_42_included(client):
    """When seed is 42, body includes seed: 42."""
    mc = ModelConfig(provider="openai", seed=42)
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        client.chat(mc, "hello")

    body = _extract_body(mock_open)
    assert body["seed"] == 42


# ── Response parsing ─────────────────────────────────────────────────────────

def test_response_text_extracted(client, model_config):
    """Response text comes from choices[0].message.content."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(content="Hello, human!")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        response_text, *_ = result[:4]
    assert response_text == "Hello, human!"


def test_tokens_extracted(client, model_config):
    """Token counts come from usage.prompt_tokens and usage.completion_tokens."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data(prompt_tokens=42,
                                                      completion_tokens=7)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = client.chat(model_config, "hello")
        _, tokens_in, tokens_out, _ = result[:4]
    assert tokens_in == 42
    assert tokens_out == 7


def test_latency_measured(client, model_config):
    """Latency is measured via time.monotonic() roundtrip."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = _fake_response_data()

    with patch("time.monotonic", side_effect=[0.0, 1.5]):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = client.chat(model_config, "hello")
            _, _, _, latency_ms = result[:4]
    assert latency_ms == 1500.0


# ── Error handling ───────────────────────────────────────────────────────────

def test_http_401_raises_runtime_error(client, model_config):
    """HTTP 401 raises RuntimeError with api_key hint."""
    mock_http_error = urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", 401, "Unauthorized", {}, None
    )
    mock_http_error.read = MagicMock(return_value=b"{}")
    with patch("urllib.request.urlopen", side_effect=mock_http_error):
        with pytest.raises(RuntimeError, match="api_key"):
            client.chat(model_config, "hello")


def test_http_404_raises_runtime_error(client, model_config):
    """HTTP 404 raises RuntimeError with model name and 'not found'."""
    mock_http_error = urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", 404, "Not Found", {}, None
    )
    mock_http_error.read = MagicMock(return_value=b"{}")
    with patch("urllib.request.urlopen", side_effect=mock_http_error):
        with pytest.raises(RuntimeError, match="not found"):
            client.chat(model_config, "hello")


def test_http_500_raises_runtime_error(client, model_config):
    """HTTP 500 raises RuntimeError with status code."""
    mock_http_error = urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", 500,
        "Internal Server Error", {}, None
    )
    with patch("urllib.request.urlopen", side_effect=mock_http_error):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.chat(model_config, "hello")


def test_connection_refused_raises_connection_error(client, model_config):
    """A URLError with ConnectionRefusedError raises stdlib ConnectionError."""
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError(
                   ConnectionRefusedError("refused"))):
        with pytest.raises(ConnectionError, match="Cannot connect"):
            client.chat(model_config, "hello")


def test_socket_timeout_raises_timeout_error(client, model_config):
    """A socket.timeout raises TimeoutError."""
    with patch("urllib.request.urlopen",
               side_effect=socket.timeout("timed out")):
        with pytest.raises(TimeoutError, match="timed out"):
            client.chat(model_config, "hello")


def test_invalid_json_raises_runtime_error(client, model_config):
    """Non-JSON response raises RuntimeError."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"not json at all!!!"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Invalid response"):
            client.chat(model_config, "hello")


def test_empty_choices_raises(client, model_config):
    """Response with empty choices list raises RuntimeError."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    # Build a response with empty choices
    mock_resp.read.return_value = json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="no choices"):
            client.chat(model_config, "hello")
