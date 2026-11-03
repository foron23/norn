"""Target surface: tools with roles, planting doors, and corpus binding."""

from __future__ import annotations

import pytest

from norn.targets.binding import bind_case, plant_request, render, slots_in
from norn.targets.manifest import Door, TargetManifest, ToolSpec, from_dict

BASE = {
    "name": "lab",
    "base_url": "http://127.0.0.1:8099",
    "chat_endpoint": "/api/chat",
    "capabilities": ["text", "stream", "tool_calls", "policy"],
}


def _manifest(**extra) -> TargetManifest:
    return from_dict({**BASE, **extra})


# ── manifest: tools ────────────────────────────────────────────────────────


def test_tools_and_doors_parse() -> None:
    manifest = _manifest(
        tools={
            "write_crm_note": {"role": "record_write", "args": ["contact", "text"]},
            "send_email": {"role": "outbound_send"},
        },
        doors=[
            {
                "id": "inbound_mail",
                "method": "post",
                "endpoint": "/webhooks/mail",
                "headers": {"X-Webhook-Token": "{{env:LAB_MAIL_TOKEN}}"},
                "body": {"from": "ops@ext.test", "body": "{{payload}}"},
            }
        ],
    )

    assert manifest.role_of("send_email") == "outbound_send"
    assert manifest.role_of("unknown_tool") is None
    assert manifest.roles() == {
        "write_crm_note": "record_write",
        "send_email": "outbound_send",
    }
    door = manifest.door("inbound_mail")
    assert isinstance(door, Door)
    assert door.method == "POST"  # normalised
    assert manifest.door("nope") is None


def test_tool_role_must_be_known() -> None:
    with pytest.raises(ValueError, match="unknown tool role"):
        ToolSpec(role="send_everything")


def test_tool_spec_rejects_empty_arg() -> None:
    with pytest.raises(ValueError, match="tool arg"):
        ToolSpec(role="outbound_send", args=[""])


def test_tool_mapping_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown field in tool"):
        _manifest(tools={"send_email": {"role": "outbound_send", "scope": "all"}})


def test_door_method_must_be_a_write() -> None:
    with pytest.raises(ValueError, match="door method"):
        _manifest(doors=[{"id": "d", "method": "GET", "endpoint": "/x"}])


def test_duplicate_door_id_is_rejected() -> None:
    door = {"id": "dup", "method": "POST", "endpoint": "/x"}
    with pytest.raises(ValueError, match="duplicate door id"):
        _manifest(doors=[door, dict(door)])


# ── binding ────────────────────────────────────────────────────────────────


def test_case_without_plant_binds_trivially() -> None:
    binding = bind_case({"content": "hello"}, _manifest())
    assert binding.usable
    assert binding.plants == ()


def test_plant_binds_a_declared_door() -> None:
    manifest = _manifest(
        doors=[{"id": "upload", "method": "POST", "endpoint": "/api/uploads", "body": {"c": "{{payload}}"}}]
    )
    binding = bind_case({"content": "doc", "plant": ["upload"]}, manifest)

    assert binding.usable
    assert [plant.door_id for plant in binding.plants] == ["upload"]
    assert binding.plants[0].body == {"c": "{{payload}}"}


def test_missing_door_drops_the_case_with_reason() -> None:
    binding = bind_case({"content": "doc", "plant": "upload"}, _manifest())
    assert not binding.usable
    assert binding.reason == "no_door:upload"


def test_requires_role_needs_a_tool_with_that_role() -> None:
    manifest = _manifest(tools={"send_email": {"role": "outbound_send"}})
    assert bind_case({"content": "x", "requires_role": "outbound_send"}, manifest).usable

    dropped = bind_case({"content": "x", "requires_role": "record_write"}, manifest)
    assert dropped.reason == "no_role:record_write"


def test_unknown_role_is_rejected_at_bind_time() -> None:
    assert bind_case({"content": "x", "requires_role": "sudo"}, _manifest()).reason == "no_role:sudo"


def test_unknown_slot_drops_the_case() -> None:
    binding = bind_case({"content": "leak {{secret}}"}, _manifest())
    assert binding.reason == "bad_slot:secret"


def test_slots_are_discovered_recursively() -> None:
    found = slots_in({"a": ["{{canary}}"], "b": {"c": "{{env:TOKEN}}"}})
    assert found == {"canary", "env:TOKEN"}


# ── rendering ──────────────────────────────────────────────────────────────


def test_render_substitutes_value_slots() -> None:
    assert render("note {{canary}}: {{payload}}", canary="CNRY-1", payload="hi") == (
        "note CNRY-1: hi"
    )


def test_render_reads_env_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NORN_TOKEN_X", "s3cret")
    assert render("{{env:NORN_TOKEN_X}}", canary="c", payload="p") == "s3cret"


def test_render_fails_loudly_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NORN_TOKEN_MISSING", raising=False)
    with pytest.raises(ValueError, match="NORN_TOKEN_MISSING"):
        render("{{env:NORN_TOKEN_MISSING}}", canary="c", payload="p")


def test_plant_request_renders_headers_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAIL_TOK", "tok")
    manifest = _manifest(
        doors=[
            {
                "id": "mail",
                "method": "POST",
                "endpoint": "/webhooks/mail",
                "headers": {"X-Webhook-Token": "{{env:MAIL_TOK}}"},
                "body": {"body": "{{payload}} — {{canary}}"},
            }
        ]
    )
    plant = bind_case({"content": "doc", "plant": "mail"}, manifest).plants[0]
    method, endpoint, headers, body = plant_request(plant, canary="CNRY-9", payload="doc")

    assert (method, endpoint) == ("POST", "/webhooks/mail")
    assert headers == {"X-Webhook-Token": "tok"}
    assert body == {"body": "doc — CNRY-9"}
