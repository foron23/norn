"""Unit tests for OllamaClient HTTP chat calls (httpx, NOR-05)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
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


def _fake_response(content: str = "test response",
                   prompt_eval: int = 10,
                   eval_count: int = 5,
                   total_duration: int = 500_000_000) -> httpx.Response:
    """Build a fake Ollama JSON response."""
    return httpx.Response(200, json={
        "model": "test",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "done": True,
    })


def _patch_client(*responses):
    """Patch httpx.Client with a fake whose post() returns ``responses`` in order."""
    fake = MagicMock()
    fake.post.side_effect = list(responses)
    patcher = patch("norn.runtime.ollama_client.httpx.Client", return_value=fake)
    return patcher, fake


def _extract_body(fake: MagicMock) -> dict:
    """Extract the JSON body passed to the last post() call."""
    return fake.post.call_args.kwargs["json"]


# ── URL / body construction ──────────────────────────────────────────────────

def test_url_uses_host_and_port(client, model_config):
    """POST is sent to http://{host}:{port}/api/chat."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    assert fake.post.call_args[0][0] == "http://localhost:11434/api/chat"


def test_content_type_header_is_json(client, model_config):
    """Request includes Content-Type: application/json header."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    headers = fake.post.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"


def test_custom_host_and_port():
    """Non-default host and port are reflected in the URL."""
    mc = ModelConfig(host="192.168.1.10", port=9999)
    client = OllamaClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert fake.post.call_args[0][0] == "http://192.168.1.10:9999/api/chat"


# ── JSON body checks ─────────────────────────────────────────────────────────

def test_body_includes_model_name(client, model_config):
    """Request body includes the model_name."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    assert _extract_body(fake)["model"] == "llama3.1:8b"


def test_body_includes_user_message(client, model_config):
    """Request body includes the user prompt in messages[0].content."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello world")
    body = _extract_body(fake)
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello world"


def test_body_includes_keep_alive(client, model_config):
    """Request body includes keep_alive: '5m'."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    assert _extract_body(fake)["keep_alive"] == "5m"


def test_body_stream_is_false(client, model_config):
    """Request body includes stream: false."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    assert _extract_body(fake)["stream"] is False


def test_body_includes_options(client, model_config):
    """Request body includes options dict with generation parameters."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    opts = _extract_body(fake)["options"]
    assert opts["temperature"] == 0.0
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 2048


def test_seed_none_coerces_to_zero(client):
    """When seed is None, the body uses 0 (Ollama random seed)."""
    mc = ModelConfig(seed=None)
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert _extract_body(fake)["options"]["seed"] == 0


def test_seed_42_preserved(client):
    """When seed is 42, the body uses 42."""
    mc = ModelConfig(seed=42)
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert _extract_body(fake)["options"]["seed"] == 42


def test_num_predict_uses_max_tokens(client):
    """num_predict in options equals max_tokens from ModelConfig."""
    mc = ModelConfig(max_tokens=512)
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert _extract_body(fake)["options"]["num_predict"] == 512


# ── Response parsing ─────────────────────────────────────────────────────────

def test_response_text_extracted_from_message_content(client, model_config):
    """Response text comes from message.content in the JSON response."""
    patcher, _ = _patch_client(_fake_response(content="Hello, human!"))
    with patcher:
        result = client.chat(model_config, "hello")
        response_text, *_ = result[:4]
    assert response_text == "Hello, human!"


def test_tokens_in_extracted_from_prompt_eval_count(client, model_config):
    """tokens_in comes from prompt_eval_count."""
    patcher, _ = _patch_client(_fake_response(prompt_eval=42))
    with patcher:
        result = client.chat(model_config, "hello")
        _, tokens_in, _, _ = result[:4]
    assert tokens_in == 42


def test_tokens_out_extracted_from_eval_count(client, model_config):
    """tokens_out comes from eval_count."""
    patcher, _ = _patch_client(_fake_response(eval_count=7))
    with patcher:
        result = client.chat(model_config, "hello")
        _, _, tokens_out, _ = result[:4]
    assert tokens_out == 7


def test_latency_computed_from_total_duration(client, model_config):
    """latency_ms = total_duration_ns / 1_000_000."""
    patcher, _ = _patch_client(_fake_response(total_duration=2_000_000_000))
    with patcher:
        result = client.chat(model_config, "hello")
        _, _, _, latency_ms = result[:4]
    assert latency_ms == 2000.0


def test_latency_is_float(client, model_config):
    """latency_ms is a float, not an int."""
    patcher, _ = _patch_client(_fake_response(total_duration=500_000_000))
    with patcher:
        result = client.chat(model_config, "hello")
        _, _, _, latency_ms = result[:4]
    assert isinstance(latency_ms, float)
    assert latency_ms == 500.0


# ── Error handling ───────────────────────────────────────────────────────────

def test_connection_failure_raises_runtime_error(client, model_config):
    """A non-connect transport error raises RuntimeError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.TransportError("Unknown host")
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake), pytest.raises(
        RuntimeError, match="Failed to connect"
    ):
        client.chat(model_config, "hello")


def test_connection_refused_raises_ollama_connection_error(client, model_config):
    """A connect error raises OllamaConnectionError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.ConnectError("connection refused")
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake), pytest.raises(
        OllamaConnectionError, match="Is Ollama running?"
    ):
        client.chat(model_config, "hello")


def test_socket_timeout_raises_timeout_error(client, model_config):
    """A read timeout raises TimeoutError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.ReadTimeout("timed out")
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake), pytest.raises(
        TimeoutError, match="timed out"
    ):
        client.chat(model_config, "hello")


def test_os_error_raises_runtime_error(client, model_config):
    """An OSError wrapped by httpx raises RuntimeError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.NetworkError("No route to host")
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake), pytest.raises(
        RuntimeError, match="Failed to connect"
    ):
        client.chat(model_config, "hello")


def test_invalid_json_raises_runtime_error(client, model_config):
    """Non-JSON response raises RuntimeError."""
    patcher, _ = _patch_client(httpx.Response(200, text="not json at all!!!"))
    with patcher, pytest.raises(RuntimeError, match="Invalid response"):
        client.chat(model_config, "hello")


def test_http_404_raises_runtime_error_with_model_name(client, model_config):
    """HTTP 404 raises RuntimeError with model name and 'not found'."""
    patcher, _ = _patch_client(httpx.Response(404, json={"error": "model not loaded"}))
    with patcher, pytest.raises(RuntimeError, match="not found on Ollama server"):
        client.chat(model_config, "hello")


def test_http_500_raises_runtime_error(client, model_config):
    """HTTP 500 raises RuntimeError with status code (after retries exhausted)."""
    patcher, _ = _patch_client(
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
    )
    with patcher, patch("norn.runtime.retry.time.sleep"), pytest.raises(
        RuntimeError, match="server error.*HTTP 500"
    ):
        client.chat(model_config, "hello")

def test_error_messages_include_host_and_port(client):
    """All error messages include host:port information."""
    mc = ModelConfig(host="192.168.1.10", port=9999)
    fake = MagicMock()
    fake.post.side_effect = httpx.ConnectError("refused")
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake), pytest.raises(
        OllamaConnectionError, match="192.168.1.10:9999"
    ):
        client.chat(mc, "hello")


# ── Timeout passthrough ──────────────────────────────────────────────────────

def test_timeout_passed_to_http_client(client, model_config):
    """The timeout kwarg is passed to httpx.Client."""
    mc = ModelConfig(timeout=30.0)
    patcher, _ = _patch_client(_fake_response())
    with patcher as mock_client_cls:
        client.chat(mc, "hello")
    assert mock_client_cls.call_args.kwargs["timeout"] == 30.0


def test_default_timeout_is_60_seconds(client, model_config):
    """Default timeout is 60.0 when not overridden."""
    patcher, _ = _patch_client(_fake_response())
    with patcher as mock_client_cls:
        client.chat(model_config, "hello")
    assert mock_client_cls.call_args.kwargs["timeout"] == 60.0


# ── list_models classmethod ──────────────────────────────────────────────────

def test_list_models_returns_model_names():
    """list_models returns list of model name strings."""
    fake = MagicMock()
    fake.get.return_value = httpx.Response(200, json={
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "mistral:7b"},
            {"name": "gemma:2b"},
        ]
    })
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake):
        models = OllamaClient.list_models("localhost", 11434)
    assert models == ["llama3.1:8b", "mistral:7b", "gemma:2b"]


def test_list_models_empty_when_no_models():
    """list_models returns empty list when server has no models."""
    fake = MagicMock()
    fake.get.return_value = httpx.Response(200, json={"models": []})
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake):
        models = OllamaClient.list_models("localhost", 11434)
    assert models == []


def test_list_models_uses_api_tags_endpoint():
    """list_models calls GET /api/tags on the configured host:port."""
    fake = MagicMock()
    fake.get.return_value = httpx.Response(200, json={"models": []})
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake) as mock_client_cls:
        OllamaClient.list_models("192.168.1.10", 9999, timeout=5.0)
    assert fake.get.call_args[0][0] == "http://192.168.1.10:9999/api/tags"
    assert mock_client_cls.call_args.kwargs["timeout"] == 5.0


def test_list_models_default_timeout_is_5_seconds():
    """list_models uses 5-second default timeout."""
    fake = MagicMock()
    fake.get.return_value = httpx.Response(200, json={"models": []})
    with patch("norn.runtime.ollama_client.httpx.Client", return_value=fake) as mock_client_cls:
        OllamaClient.list_models("localhost", 11434)
    assert mock_client_cls.call_args.kwargs["timeout"] == 5.0


def test_ollama_connection_error_is_connection_error_subclass():
    """OllamaConnectionError inherits from ConnectionError."""
    assert issubclass(OllamaConnectionError, ConnectionError)


# ── Retry / backoff (NOR-05) ─────────────────────────────────────────────────

def test_retry_429_then_success(client, model_config):
    """429 then 200 → request succeeds after a retry."""
    patcher, fake = _patch_client(
        httpx.Response(429),
        _fake_response(content="ok after retry"),
    )
    with patcher, patch("norn.runtime.retry.time.sleep"):
        result = client.chat(model_config, "hello")
    assert result[0] == "ok after retry"
    assert fake.post.call_count == 2


def test_retry_503_thrice_raises_runtime_error(client, model_config):
    """503 x3 → RuntimeError with HTTP status, after 3 attempts."""
    patcher, fake = _patch_client(
        httpx.Response(503), httpx.Response(503), httpx.Response(503)
    )
    with patcher, patch("norn.runtime.retry.time.sleep"), pytest.raises(
        RuntimeError, match="HTTP 503"
    ):
        client.chat(model_config, "hello")
    assert fake.post.call_count == 3
