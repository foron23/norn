"""Unit tests for the OpenCode relay driver (NOR-30)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from norn.domain.models import ModelConfig
from norn.runtime.opencode_client import (
    ANTHROPIC_MESSAGES,
    CHAT_COMPLETIONS,
    CODEX_RESPONSES,
    OpenCodeRelayClient,
    diagnose_error,
    model_transport,
    opencode_session_id,
)
from norn.runtime.providers import build_provider

GO_BASE = "https://opencode.ai/zen/go/v1"
SESSION = "x-opencode-session"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def go_config() -> ModelConfig:
    """Campaign config aimed at the Go relay with an explicit key."""
    return ModelConfig(
        provider="opencode-go",
        base_url=GO_BASE,
        model_name="deepseek-v4.1-flash",
        api_key="relay-key",
    )


# ── Utilities ────────────────────────────────────────────────────────────────

def _fake_response(content: str = "OK") -> httpx.Response:
    """Build a fake chat completions success response."""
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    })


def _patch_client(*responses):
    """Patch httpx.Client so post() returns ``responses`` in order."""
    fake = MagicMock()
    fake.post.side_effect = list(responses)
    patcher = patch("norn.runtime.openai_client.httpx.Client", return_value=fake)
    return patcher, fake


def _headers(fake: MagicMock) -> dict:
    """Headers passed to the last post() call."""
    return fake.post.call_args.kwargs["headers"]


# ── Session identity ─────────────────────────────────────────────────────────

def test_session_id_is_stable_and_opaque():
    """The same parts always produce the same 32-char hex session."""
    first = opencode_session_id(7, "L1_AT_12_refusal_ban_1", 0)
    second = opencode_session_id(7, "L1_AT_12_refusal_ban_1", 0)
    assert first == second
    assert len(first) == 32
    assert all(c in "0123456789abcdef" for c in first)


def test_session_id_differs_per_replica():
    """Replicas of the same case get different sessions."""
    assert opencode_session_id(7, "case", 0) != opencode_session_id(7, "case", 1)


def test_session_id_ignores_empty_parts():
    """Optional scope arguments can be None without changing the id shape."""
    assert opencode_session_id(7, None, "case") == opencode_session_id(7, "", "case")


def test_client_session_is_used_for_every_request(go_config):
    """A client-level session id is sent as x-opencode-session."""
    client = OpenCodeRelayClient(session_id="campaign-scope")
    patcher, fake = _patch_client(_fake_response(), _fake_response())
    with patcher:
        client.chat(go_config, "hello")
        client.chat(go_config, "again")
    assert fake.post.call_count == 2
    for call in fake.post.call_args_list:
        assert call.kwargs["headers"][SESSION] == "campaign-scope"


def test_explicit_session_in_extra_headers_wins(go_config):
    """A campaign that pins its own session overrides the driver default."""
    go_config.extra_headers = {SESSION: "pinned-by-config"}
    client = OpenCodeRelayClient(session_id="client-scope")
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(go_config, "hello")
    assert _headers(fake)[SESSION] == "pinned-by-config"


# ── Headers ──────────────────────────────────────────────────────────────────

def test_relay_headers_are_sent(go_config):
    """Every relay request carries the identifying UA, session and bearer key."""
    client = OpenCodeRelayClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(go_config, "hello")
    headers = _headers(fake)
    assert headers["User-Agent"] == "norn/0.1.0 (llm-redteam)"
    assert headers[SESSION] == client.resolve_session(go_config)
    assert headers["Authorization"] == "Bearer relay-key"


def test_extra_headers_are_forwarded(go_config):
    """Vendor headers from the campaign reach the wire."""
    go_config.extra_headers = {"X-Norn-Test": "1"}
    client = OpenCodeRelayClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(go_config, "hello")
    assert _headers(fake)["X-Norn-Test"] == "1"


def test_authorization_wins_over_extra_headers(go_config):
    """A stray Authorization header never replaces the configured api_key."""
    go_config.extra_headers = {"Authorization": "Bearer wrong"}
    client = OpenCodeRelayClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(go_config, "hello")
    assert _headers(fake)["Authorization"] == "Bearer relay-key"


def test_relay_config_does_not_mutate_input(go_config):
    """Header injection works on a copy, so the campaign config stays clean."""
    client = OpenCodeRelayClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher:
        client.chat(go_config, "hello")
    assert go_config.extra_headers == {}
    assert fake.post.call_count == 1


# ── Transport guard ──────────────────────────────────────────────────────────

def test_deepseek_uses_chat_completions():
    """The model family we measure is served by the supported transport."""
    assert model_transport("opencode-go", "deepseek-v4.1-flash") == CHAT_COMPLETIONS
    assert model_transport("opencode-zen", "glm-5.3") == CHAT_COMPLETIONS
    assert model_transport("opencode-go", "kimi-k3") == CHAT_COMPLETIONS


def test_claude_on_zen_needs_anthropic_messages():
    """claude-* is Messages-only on Zen."""
    assert model_transport("opencode-zen", "claude-haiku-5-5") == ANTHROPIC_MESSAGES


def test_qwen_and_minimax_need_anthropic_messages():
    """qwen* is Messages on both families; minimax-* on Go."""
    assert model_transport("opencode-go", "qwen3.8-max") == ANTHROPIC_MESSAGES
    assert model_transport("opencode-go", "minimax-m3") == ANTHROPIC_MESSAGES


def test_gpt_and_grok_need_codex_responses():
    """GPT/Grok/Muse Spark are Responses-only on both families."""
    assert model_transport("opencode-go", "gpt-6-luna") == CODEX_RESPONSES
    assert model_transport("opencode-zen", "grok-4.7") == CODEX_RESPONSES
    assert model_transport("opencode-zen", "muse-spark-1.3-contributor") == CODEX_RESPONSES


def test_unsupported_transport_fails_before_http(go_config):
    """A Messages-only model fails with a named transport and no request sent."""
    go_config.model_name = "claude-haiku-5-5"
    go_config.provider = "opencode-zen"
    client = OpenCodeRelayClient()
    patcher, fake = _patch_client(_fake_response())
    with patcher, pytest.raises(RuntimeError, match="anthropic_messages"):
        client.chat(go_config, "hello")
    assert fake.post.call_count == 0


# ── Error diagnostics ────────────────────────────────────────────────────────

def test_diagnose_free_tier_exhausted():
    """The anonymous free tier message points at the subscription."""
    hint = diagnose_error(429, '{"error":{"type":"FreeUsageLimitError"}}')
    assert hint is not None and "free tier" in hint


def test_diagnose_missing_session():
    """A missing session header is reported as such, not as a generic 400."""
    hint = diagnose_error(400, '{"error":{"type":"MissingSessionID"}}')
    assert hint is not None and "x-opencode-session" in hint


def test_diagnose_go_requires_credential():
    """The credential error names plan and region, the two real causes."""
    hint = diagnose_error(401, '{"message":"Upstream request failed: Go requires a credential"}')
    assert hint is not None and "region" in hint


def test_diagnose_cloudflare_1010():
    """Cloudflare 1010 is a User-Agent problem, not an auth problem."""
    hint = diagnose_error(403, "error code: 1010")
    assert hint is not None and "User-Agent" in hint


def test_diagnose_ignores_unrelated_errors():
    """Errors with no relay-specific cause keep the client's own message."""
    assert diagnose_error(500, "boom") is None


def test_relay_error_is_enriched(go_config):
    """A relay-specific failure appends the hint to the client's message."""
    client = OpenCodeRelayClient()
    patcher, _ = _patch_client(
        httpx.Response(401, json={"message": "Upstream request failed: Go requires a credential"})
    )
    with patcher, pytest.raises(RuntimeError) as excinfo:
        client.chat(go_config, "hello")
    message = str(excinfo.value)
    assert "401" in message
    assert "relay:" in message
    assert "region" in message


def test_plain_401_keeps_generic_hint(go_config):
    """A 401 without a relay payload still points at the api_key."""
    client = OpenCodeRelayClient()
    patcher, _ = _patch_client(httpx.Response(401, json={"error": "unauthorized"}))
    with patcher, pytest.raises(RuntimeError, match="api_key"):
        client.chat(go_config, "hello")


# ── Provider registry and credential resolution ──────────────────────────────

def test_build_provider_returns_relay_client():
    """Both relay names build the relay driver."""
    assert isinstance(build_provider("opencode-go"), OpenCodeRelayClient)
    assert isinstance(build_provider("opencode-zen"), OpenCodeRelayClient)
    assert isinstance(build_provider("OPENCODE-GO"), OpenCodeRelayClient)


def test_unknown_provider_lists_relays():
    """The factory error advertises the new providers."""
    with pytest.raises(ValueError, match="opencode-go"):
        build_provider("wat")


def test_relay_key_comes_from_its_own_env_var(monkeypatch):
    """opencode-go resolves OPENCODE_GO_API_KEY, not the Ollama variable."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-key")
    assert ModelConfig(provider="opencode-go").api_key == "go-key"


def test_zen_key_comes_from_its_own_env_var(monkeypatch):
    """opencode-zen resolves OPENCODE_ZEN_API_KEY."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")
    assert ModelConfig(provider="opencode-zen").api_key == "zen-key"


def test_other_providers_keep_ollama_env(monkeypatch):
    """The relay resolution is scoped: openai still reads OLLAMA_API_KEY."""
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    assert ModelConfig(provider="openai").api_key == "ollama-key"
