"""Unit tests for OpenAICompatibleClient HTTP chat calls (httpx, NOR-05)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
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

def _fake_response(content: str = "test response",
                   prompt_tokens: int = 10,
                   completion_tokens: int = 5) -> httpx.Response:
    """Build a fake OpenAI Chat Completions JSON response."""
    return httpx.Response(200, json={
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
    })


def _patch_client(*responses):
    """Patch httpx.Client with a fake whose post() returns ``responses`` in order."""
    fake = MagicMock()
    fake.post.side_effect = list(responses)
    patcher = patch("norn.runtime.openai_client.httpx.Client", return_value=fake)
    return patcher, fake


def _extract_body(fake: MagicMock) -> dict:
    """Extract the JSON body passed to the last post() call."""
    return fake.post.call_args.kwargs["json"]


# ── URL / header construction ────────────────────────────────────────────────

def test_url_uses_base_url(client, model_config):
    """POST is sent to {base_url}/chat/completions."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    assert fake.post.call_args[0][0] == "https://api.openai.com/v1/chat/completions"


def test_custom_base_url():
    """Non-default base_url is reflected in the URL."""
    mc = ModelConfig(provider="openai", base_url="http://localhost:8080/v1")
    client = OpenAICompatibleClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert fake.post.call_args[0][0] == "http://localhost:8080/v1/chat/completions"


def test_trailing_slash_stripped():
    """base_url with trailing slash produces single /chat/completions."""
    mc = ModelConfig(provider="openai", base_url="http://localhost:8080/v1/")
    client = OpenAICompatibleClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert fake.post.call_args[0][0] == "http://localhost:8080/v1/chat/completions"
    assert "//chat" not in fake.post.call_args[0][0]


def test_content_type_header_is_json(client, model_config):
    """Request includes Content-Type: application/json header."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    headers = fake.post.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"


def test_auth_header_when_api_key_set():
    """Authorization: Bearer *** sent when api_key is set."""
    mc = ModelConfig(provider="openai", api_key="sk-test-1234")
    client = OpenAICompatibleClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    headers = fake.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-test-1234"


def test_no_auth_header_when_api_key_none(client, model_config):
    """No Authorization header when api_key is None."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    headers = fake.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


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


def test_body_includes_temperature_top_p_max_tokens(client, model_config):
    """Request body includes temperature, top_p, and max_tokens as flat keys."""
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(model_config, "hello")
    body = _extract_body(fake)
    assert body["temperature"] == 0.0
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 2048


def test_seed_none_omitted(client):
    """When seed is None, seed key is not in request body."""
    mc = ModelConfig(provider="openai", seed=None)
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert "seed" not in _extract_body(fake)


def test_seed_42_included(client):
    """When seed is 42, body includes seed: 42."""
    mc = ModelConfig(provider="openai", seed=42)
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(mc, "hello")
    assert _extract_body(fake)["seed"] == 42


# ── Response parsing ─────────────────────────────────────────────────────────

def test_response_text_extracted(client, model_config):
    """Response text comes from choices[0].message.content."""
    patcher, _ = _patch_client(_fake_response(content="Hello, human!"))
    with patcher:
        result = client.chat(model_config, "hello")
        response_text, *_ = result[:4]
    assert response_text == "Hello, human!"


def test_tokens_extracted(client, model_config):
    """Token counts come from usage.prompt_tokens and usage.completion_tokens."""
    patcher, _ = _patch_client(_fake_response(prompt_tokens=42, completion_tokens=7))
    with patcher:
        result = client.chat(model_config, "hello")
        _, tokens_in, tokens_out, _ = result[:4]
    assert tokens_in == 42
    assert tokens_out == 7


def test_latency_measured(client, model_config):
    """Latency is measured via time.monotonic() roundtrip."""
    patcher, _ = _patch_client(_fake_response())
    with patch("time.monotonic", side_effect=[0.0, 1.5]), patcher:
        result = client.chat(model_config, "hello")
        _, _, _, latency_ms = result[:4]
    assert latency_ms == 1500.0


# ── Error handling ───────────────────────────────────────────────────────────

def test_http_401_raises_runtime_error(client, model_config):
    """HTTP 401 raises RuntimeError with api_key hint."""
    patcher, _ = _patch_client(httpx.Response(401, json={"error": "unauthorized"}))
    with patcher, pytest.raises(RuntimeError, match="api_key"):
        client.chat(model_config, "hello")


def test_http_404_raises_runtime_error(client, model_config):
    """HTTP 404 raises RuntimeError with model name and 'not found'."""
    patcher, _ = _patch_client(httpx.Response(404, json={"error": "nope"}))
    with patcher, pytest.raises(RuntimeError, match="not found"):
        client.chat(model_config, "hello")


def test_http_500_raises_runtime_error(client, model_config):
    """HTTP 500 raises RuntimeError with status code (after retries exhausted)."""
    patcher, _ = _patch_client(
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(500, json={"error": "boom"}),
    )
    with patcher, patch("norn.runtime.retry.time.sleep"), pytest.raises(
        RuntimeError, match="HTTP 500"
    ):
        client.chat(model_config, "hello")

def test_connection_refused_raises_connection_error(client, model_config):
    """A connect error raises stdlib ConnectionError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.ConnectError("connection refused")
    with patch("norn.runtime.openai_client.httpx.Client", return_value=fake), pytest.raises(
        ConnectionError, match="Cannot connect"
    ):
        client.chat(model_config, "hello")


def test_socket_timeout_raises_timeout_error(client, model_config):
    """A read timeout raises TimeoutError."""
    fake = MagicMock()
    fake.post.side_effect = httpx.ReadTimeout("timed out")
    with patch("norn.runtime.openai_client.httpx.Client", return_value=fake), pytest.raises(
        TimeoutError, match="timed out"
    ):
        client.chat(model_config, "hello")


def test_invalid_json_raises_runtime_error(client, model_config):
    """Non-JSON response raises RuntimeError."""
    patcher, _ = _patch_client(httpx.Response(200, text="not json at all!!!"))
    with patcher, pytest.raises(RuntimeError, match="Invalid response"):
        client.chat(model_config, "hello")


def test_empty_choices_raises(client, model_config):
    """Response with empty choices list raises RuntimeError."""
    patcher, _ = _patch_client(httpx.Response(200, json={
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
    }))
    with patcher, pytest.raises(RuntimeError, match="no choices"):
        client.chat(model_config, "hello")


# ── Timeout passthrough ──────────────────────────────────────────────────────

def test_timeout_passed_to_http_client():
    """The timeout kwarg is passed to httpx.Client."""
    mc = ModelConfig(provider="openai", timeout=30.0)
    client = OpenAICompatibleClient()
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


# ── Retry / backoff (NOR-05) ─────────────────────────────────────────────────

def test_retry_429_twice_then_success(client, model_config):
    """429 twice then 200 → request succeeds after retries."""
    patcher, fake = _patch_client(
        httpx.Response(429),
        httpx.Response(429),
        _fake_response(content="ok after retries"),
    )
    with patcher, patch("norn.runtime.retry.time.sleep"):
        result = client.chat(model_config, "hello")
    assert result[0] == "ok after retries"
    assert fake.post.call_count == 3


def test_retry_500_thrice_raises_runtime_error(client, model_config):
    """500 x3 → RuntimeError with HTTP status, after 3 attempts."""
    patcher, fake = _patch_client(
        httpx.Response(500), httpx.Response(500), httpx.Response(500)
    )
    with patcher, patch("norn.runtime.retry.time.sleep"), pytest.raises(
        RuntimeError, match="HTTP 500"
    ):
        client.chat(model_config, "hello")
    assert fake.post.call_count == 3


def test_retry_backoff_delay_increases(client, model_config):
    """Backoff grows across attempts: 1s-ish, then 2s-ish, then 4s-ish."""
    patcher, _ = _patch_client(
        httpx.Response(429), httpx.Response(429), _fake_response()
    )
    with patcher, patch("norn.runtime.retry.time.sleep") as mock_sleep:
        client.chat(model_config, "hello")
    delays = [c.args[0] for c in mock_sleep.call_args_list]
    assert len(delays) == 2
    assert delays[0] < delays[1]


def test_retry_after_header_respected(client, model_config):
    """Retry-After header overrides the backoff delay."""
    patcher, _ = _patch_client(
        httpx.Response(429, headers={"Retry-After": "3.0"}),
        _fake_response(),
    )
    with patcher, patch("norn.runtime.retry.time.sleep") as mock_sleep:
        client.chat(model_config, "hello")
    delay = mock_sleep.call_args[0][0]
    assert delay >= 3.0


def test_401_not_retried(client, model_config):
    """Non-retryable statuses are not retried."""
    patcher, fake = _patch_client(httpx.Response(401, json={"error": "bad"}))
    with patcher, pytest.raises(RuntimeError, match="api_key"):
        client.chat(model_config, "hello")
    assert fake.post.call_count == 1


def test_retry_disabled_with_single_attempt(client, model_config):
    """retry_attempts=1 disables retries."""
    single = OpenAICompatibleClient(retry_attempts=1)
    patcher, fake = _patch_client(httpx.Response(429), _fake_response())
    with patcher, pytest.raises(RuntimeError, match="HTTP 429"):
        single.chat(model_config, "hello")
    assert fake.post.call_count == 1
