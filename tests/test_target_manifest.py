"""Tests for the target manifest schema and its YAML loader."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from norn.targets.manifest import (
    CANONICAL_EVENTS,
    CAPABILITIES,
    Predicate,
    TargetManifest,
    from_dict,
    load_manifest,
)

VALID = {
    "name": "lab-agent",
    "base_url": "http://127.0.0.1:8000",
    "chat_endpoint": "/api/chat",
    "capabilities": ["text", "stream", "tool_calls", "policy"],
    "predicates": [
        {
            "id": "note-planted",
            "method": "get",
            "endpoint": "/api/crm/notes",
            "json_path": "$.notes[*].text",
            "contains": "{{canary}}",
        }
    ],
    "deadline_ms": 1500,
    "event_map": {"run.started": "start"},
}


def test_valid_manifest_keeps_every_field() -> None:
    manifest = from_dict(VALID)

    assert isinstance(manifest, TargetManifest)
    assert manifest.name == "lab-agent"
    assert manifest.base_url == "http://127.0.0.1:8000"
    assert manifest.chat_endpoint == "/api/chat"
    assert manifest.capabilities == frozenset({"text", "stream", "tool_calls", "policy"})
    assert manifest.deadline_ms == 1500
    assert manifest.token_env == "NORN_TARGET_TOKEN"
    assert manifest.event_map == {"run.started": "start"}
    assert len(manifest.predicates) == 1
    predicate = manifest.predicates[0]
    assert predicate.id == "note-planted"
    assert predicate.method == "GET"  # normalised
    assert predicate.json_path == "$.notes[*].text"


def test_capabilities_and_events_vocabulary_are_closed() -> None:
    assert CAPABILITIES == frozenset({"text", "stream", "tool_calls", "policy", "retrieval"})
    assert "policy.decision" in CANONICAL_EVENTS
    assert len(CANONICAL_EVENTS) == 7


def test_unknown_capability_is_rejected() -> None:
    data = dict(VALID, capabilities=["text", "telepathy"])

    with pytest.raises(ValueError, match="telepathy"):
        from_dict(data)


def test_capability_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="capabilities must be a list"):
        from_dict(dict(VALID, capabilities="text"))


def test_unknown_manifest_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown manifest field"):
        from_dict(dict(VALID, descripton="typo"))


def test_missing_required_field_is_rejected() -> None:
    data = {k: v for k, v in VALID.items() if k != "chat_endpoint"}

    with pytest.raises(ValueError, match="missing manifest field: chat_endpoint"):
        from_dict(data)


def test_base_url_must_be_http_or_https() -> None:
    with pytest.raises(ValueError, match="base_url must start with http"):
        from_dict(dict(VALID, base_url="ftp://127.0.0.1:8000"))


def test_deadline_must_be_positive() -> None:
    with pytest.raises(ValueError, match="deadline_ms must be > 0"):
        from_dict(dict(VALID, deadline_ms=0))

    with pytest.raises(ValueError, match="deadline_ms must be an integer"):
        from_dict(dict(VALID, deadline_ms="8000"))


def test_event_map_only_maps_canonical_names() -> None:
    with pytest.raises(ValueError, match="unknown event in event_map"):
        from_dict(dict(VALID, event_map={"ran.away": "start"}))

    with pytest.raises(ValueError, match="event_map"):
        from_dict(dict(VALID, event_map={"run.started": ""}))


def test_predicate_requires_every_field() -> None:
    broken = [{"id": "x", "method": "GET", "endpoint": "/e", "json_path": "$"}]

    with pytest.raises(ValueError, match="missing field: contains"):
        from_dict(dict(VALID, predicates=broken))


def test_predicate_method_is_restricted_to_get_and_post() -> None:
    with pytest.raises(ValueError, match="predicate method must be one of"):
        Predicate(id="p", method="DELETE", endpoint="/e", json_path="$", contains="x")


def test_manifest_is_frozen_and_defensive_copies() -> None:
    event_map = {"run.started": "start"}
    manifest = from_dict(dict(VALID, event_map=event_map))
    event_map["run.finished"] = "end"

    assert manifest.event_map == {"run.started": "start"}
    with pytest.raises(FrozenInstanceError):
        manifest.name = "other"  # type: ignore[misc]


def test_predicates_default_to_empty_tuple() -> None:
    data = {k: v for k, v in VALID.items() if k != "predicates"}

    assert from_dict(data).predicates == ()


def test_load_manifest_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "target.yaml"
    path.write_text(
        "name: fake\n"
        "base_url: http://127.0.0.1:8099\n"
        "chat_endpoint: /api/chat\n"
        "capabilities: [text, stream]\n",
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest.name == "fake"
    assert manifest.capabilities == frozenset({"text", "stream"})
    assert manifest.deadline_ms == 8000


def test_fixture_manifest_loads() -> None:
    path = Path(__file__).parent / "fixtures" / "fake.yaml"

    manifest = load_manifest(path)

    assert manifest.name == "fake"
    assert "policy" in manifest.capabilities
    assert [p.id for p in manifest.predicates] == [
        "notes_carry_canary",
        "outbox_carries_canary",
    ]


def test_empty_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty manifest"):
        load_manifest(path)
