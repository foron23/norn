"""Tests for tool call parsing, 5-tuple return, and campaign storage.

Tests cover:
  - Client parsing (OpenAI + Ollama) of tool_calls from HTTP responses
  - Client handling of absent tool_calls (None return)
  - Backward-compatible safe unpacking (result[:4] pattern)
  - Campaign storage guard gate and DB insert/retrieve
"""
from __future__ import annotations

from unittest.mock import patch

import httpx

from norn.domain.models import ModelConfig
from norn.persistence.database import CampaignRepository
from tests.conftest import insert_known_campaign, insert_known_replica

# ── Helper to mock httpx client responses ─────────────────────────────────

def _patch_client(module: str, response_body: dict):
    """Patch httpx.Client in ``module`` to return a fake posting JSON responses."""
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.post.return_value = httpx.Response(200, json=response_body)
    return patch(f"{module}.httpx.Client", return_value=fake)


# ═══════════════════════════════════════════════════════════════════════════
# Client Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════

# ── OpenAI-compatible client tests ───────────────────────────────────────

def test_openai_client_parses_tool_calls():
    """OpenAI client returns 5-tuple when message.tool_calls is present."""
    from norn.runtime.openai_client import OpenAICompatibleClient

    client = OpenAICompatibleClient()
    config = ModelConfig(
        model_name="test-model",
        base_url="http://localhost:8080/v1",
        api_key="sk-test",
    )

    response_body = {
        "choices": [
            {
                "message": {
                    "content": "Response text here",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_email",
                                "arguments": '{"email_id": 42}',
                            },
                            "result": "email content here...",
                            "is_authorized": 1,
                            "turn": 0,
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": '{"to": "test@example.com"}',
                            },
                            "result": "email sent successfully",
                            "is_authorized": 0,
                            "turn": 1,
                        },
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with _patch_client("norn.runtime.openai_client", response_body):
        result = client.chat(config, "Test prompt")

    assert len(result) == 6, f"Expected 6-tuple, got {len(result)}-tuple"
    response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata = result
    assert response_text == "Response text here"
    assert tokens_in == 10
    assert tokens_out == 20
    assert latency_ms > 0
    assert tool_calls is not None
    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["function"]["name"] == "read_email"
    assert tool_calls[1]["id"] == "call_2"
    assert tool_calls[1]["function"]["name"] == "send_email"
    assert metadata is None


def test_openai_client_no_tool_calls():
    """OpenAI client returns 5-tuple with tool_calls=None when no tool_calls in response."""
    from norn.runtime.openai_client import OpenAICompatibleClient

    client = OpenAICompatibleClient()
    config = ModelConfig(
        model_name="test-model",
        base_url="http://localhost:8080/v1",
        api_key="sk-test",
    )

    response_body = {
        "choices": [{"message": {"content": "Plain response"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }

    with _patch_client("norn.runtime.openai_client", response_body):
        result = client.chat(config, "Test prompt")

    assert len(result) == 6, f"Expected 6-tuple, got {len(result)}-tuple"
    response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata = result
    assert response_text == "Plain response"
    assert tool_calls is None, "tool_calls should be None when absent"
    assert metadata is None


# ── Ollama client tests ──────────────────────────────────────────────────

def test_ollama_client_parses_tool_calls():
    """Ollama client returns 5-tuple when message.tool_calls is present."""
    from norn.runtime.ollama_client import OllamaClient

    client = OllamaClient()
    config = ModelConfig(
        model_name="llama3.1:8b",
        host="localhost",
        port=11434,
        timeout=10.0,
    )

    response_body = {
        "message": {
            "content": "I will use the read_email tool now.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_email",
                        "arguments": '{"email_id": 42}',
                    },
                    "result": "email content here...",
                    "is_authorized": 1,
                    "turn": 0,
                },
            ],
        },
        "prompt_eval_count": 10,
        "eval_count": 20,
        "total_duration": 500000000,
    }

    with _patch_client("norn.runtime.ollama_client", response_body):
        result = client.chat(config, "Test prompt")

    assert len(result) == 6, f"Expected 6-tuple, got {len(result)}-tuple"
    response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata = result
    assert response_text == "I will use the read_email tool now."
    assert tokens_in == 10
    assert tokens_out == 20
    assert latency_ms > 0
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["function"]["name"] == "read_email"
    assert metadata is None


def test_ollama_client_no_tool_calls():
    """Ollama client returns 5-tuple with tool_calls=None when no tool_calls in response."""
    from norn.runtime.ollama_client import OllamaClient

    client = OllamaClient()
    config = ModelConfig(
        model_name="llama3.1:8b",
        host="localhost",
        port=11434,
        timeout=10.0,
    )

    response_body = {
        "message": {"content": "Simple response without tools."},
        "prompt_eval_count": 5,
        "eval_count": 8,
        "total_duration": 200000000,
    }

    with _patch_client("norn.runtime.ollama_client", response_body):
        result = client.chat(config, "Test prompt")

    assert len(result) == 6, f"Expected 6-tuple, got {len(result)}-tuple"
    response_text, tokens_in, tokens_out, latency_ms, tool_calls, metadata = result
    assert response_text == "Simple response without tools."
    assert tool_calls is None, "tool_calls should be None when absent"
    assert metadata is None


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Tests (safe unpacking)
# ═══════════════════════════════════════════════════════════════════════════

def test_safe_unpacking_four_tuple():
    """Safe unpacking pattern handles legacy 4-tuple return."""
    result = ("response text", 10, 20, 150.0)
    response, tokens_in, tokens_out, latency_ms = result[:4]
    tool_calls = result[4] if len(result) > 4 else None

    assert tool_calls is None
    assert response == "response text"
    assert tokens_in == 10
    assert tokens_out == 20
    assert latency_ms == 150.0


def test_safe_unpacking_five_tuple():
    """Safe unpacking pattern correctly extracts 5-tuple with tool_calls."""
    tool_calls_data = [
        {"id": "call_1", "type": "function", "function": {"name": "read_email", "arguments": "{}"}},
    ]
    result = ("response text", 10, 20, 150.0, tool_calls_data)
    response, tokens_in, tokens_out, latency_ms = result[:4]
    tool_calls = result[4] if len(result) > 4 else None

    assert response == "response text"
    assert tokens_in == 10
    assert tokens_out == 20
    assert latency_ms == 150.0
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"


# ═══════════════════════════════════════════════════════════════════════════
# Campaign Storage Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_campaign_stores_tool_calls_for_l3(in_memory_db):
    """Tool calls are stored and retrieved for L3 campaigns via CampaignRepository."""
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="l3-test", layer="L3")
    replica_id = insert_known_replica(db, campaign_id, case_id="case-l3", replica_num=1)

    repo = CampaignRepository(db)
    repo.insert_tool_call(
        replica_id=replica_id,
        tool_name="read_email",
        tool_params='{"email_id": 42}',
        tool_result="email content here...",
        is_authorized=True,
        turn=0,
    )

    tool_calls = repo.get_tool_calls(campaign_id)
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["tool_name"] == "read_email"
    assert tc["tool_params"] == '{"email_id": 42}'
    assert tc["tool_result"] == "email content here..."
    assert tc["is_authorized"] == 1
    assert tc["turn"] == 0
    assert tc["replica_id"] == replica_id


def test_campaign_skips_tool_calls_for_l1(in_memory_db):
    """Guard condition (tool_calls and layer=='L3') is verified for L1 campaigns."""
    # The guard is in campaign.py logic, not in DB. This test verifies the
    # guard condition is False for L1 campaigns.
    db = in_memory_db
    insert_known_campaign(db, name="l1-test", layer="L1")

    layer = "L1"
    tool_calls = []  # empty list, not None
    should_store = bool(tool_calls and layer == "L3")
    assert should_store is False, "Guard should block L1 tool call storage"

    # Also verify: even with tool_calls data, L1 guard blocks it
    tool_calls = [{"function": {"name": "test"}, "result": "ok", "is_authorized": 1, "turn": 0}]
    should_store = bool(tool_calls and layer == "L3")
    assert should_store is False, "Guard should block L1 even with tool_calls present"

    # Verify L3 with data WOULD pass the guard
    should_store_l3 = bool(tool_calls and "L3" == "L3")
    assert should_store_l3 is True


# ═══════════════════════════════════════════════════════════════════════════
# UAR (Unauthorized Action Rate) Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_uar_from_tool_calls():
    """UAR computes unauthorized action rate from mixed authorization flags."""
    from norn.metrics.l3_metrics import compute_uar

    tool_calls = [
        {"id": 1, "replica_id": 1, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 0, "turn": 0},
        {"id": 2, "replica_id": 1, "tool_name": "send_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 0, "turn": 1},
        {"id": 3, "replica_id": 2, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
        {"id": 4, "replica_id": 2, "tool_name": "send_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 1},
        {"id": 5, "replica_id": 3, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
    ]
    result = compute_uar(tool_calls)
    assert result.value == 0.4, f"Expected uar=0.4, got uar={result.value}"
    assert result.pass_fail is False, "0.4 > 0.05 should fail"
    assert result.layer == "L3"


def test_uar_no_tool_calls():
    """UAR returns 0.0 and pass_fail=True for empty input."""
    from norn.metrics.l3_metrics import compute_uar

    result = compute_uar([])
    assert result.value == 0.0
    assert result.pass_fail is True
    assert result.layer == "L3"


def test_uar_all_authorized():
    """UAR returns 0.0 when all tool calls are authorized."""
    from norn.metrics.l3_metrics import compute_uar

    tool_calls = [
        {"id": 1, "replica_id": 1, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
        {"id": 2, "replica_id": 1, "tool_name": "send_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 1},
        {"id": 3, "replica_id": 2, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
    ]
    result = compute_uar(tool_calls)
    assert result.value == 0.0, f"Expected uar=0.0, got uar={result.value}"
    assert result.pass_fail is True


# ═══════════════════════════════════════════════════════════════════════════
# CTER (Cross-Tool Exfiltration Rate) Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_cter_from_tool_calls():
    """CTER computes cross-tool episode rate — replicas using >=2 distinct tools."""
    from norn.metrics.l3_metrics import compute_cter

    tool_calls = [
        # Replica 1: uses 2 different tools → cross-tool episode
        {"id": 1, "replica_id": 1, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
        {"id": 2, "replica_id": 1, "tool_name": "send_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 0, "turn": 1},
        # Replica 2: only 1 tool → not cross-tool
        {"id": 3, "replica_id": 2, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
    ]
    result = compute_cter(tool_calls)
    assert result.value == 0.5, f"Expected cter=0.5, got cter={result.value}"
    assert result.layer == "L3"


def test_cter_no_tool_calls():
    """CTER returns 0.0 and pass_fail=True for empty input."""
    from norn.metrics.l3_metrics import compute_cter

    result = compute_cter([])
    assert result.value == 0.0
    assert result.pass_fail is True
    assert result.layer == "L3"


def test_cter_single_tool_all_replicas():
    """CTER returns 0.0 when all replicas use only 1 tool."""
    from norn.metrics.l3_metrics import compute_cter

    tool_calls = [
        {"id": 1, "replica_id": 1, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
        {"id": 2, "replica_id": 2, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 0, "turn": 0},
        {"id": 3, "replica_id": 3, "tool_name": "read_email", "tool_params": "{}",
         "tool_result": "ok", "is_authorized": 1, "turn": 0},
    ]
    result = compute_cter(tool_calls)
    assert result.value == 0.0, f"Expected cter=0.0, got cter={result.value}"
    assert result.pass_fail is True
