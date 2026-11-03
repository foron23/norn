"""Tests for the evidence probes over the target's GET endpoints."""
from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from norn.targets.manifest import from_dict
from norn.targets.observe import (
    HIT,
    INCONCLUSIVE,
    MISS,
    Evidence,
    observe,
    select,
)
from tests.fixtures.fake_target_app import FakeTargetApp


def manifest_with(app: FakeTargetApp, **overrides):
    data = {
        "name": "fake",
        "base_url": app.base_url,
        "chat_endpoint": "/attack",
        "capabilities": ["text", "stream", "tool_calls", "policy"],
        "deadline_ms": 1500,
        "predicates": [
            {
                "id": "note-planted",
                "method": "GET",
                "endpoint": "/api/crm/notes",
                "json_path": "$.notes[*].text",
                "contains": "{{canary}}",
            }
        ],
    }
    data.update(overrides)
    return from_dict(data)


@pytest.fixture
def app() -> Iterator[FakeTargetApp]:
    with FakeTargetApp() as fake:
        yield fake


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("$.notes[*].text", ["a", "b"]),
        ("$[*].name", ["x", "y"]),
        ("$.items[1].body", ["second"]),
        ("$.missing[*].text", []),
    ],
)
def test_select_handles_the_manifest_grammar(path: str, expected: list[str]) -> None:
    body = {
        "root": "root",
        "notes": [{"text": "a"}, {"text": "b"}],
        "items": [{"name": "x"}, {"name": "y", "body": "second"}],
    }
    if path == "$[*].name":
        body = [{"name": "x"}, {"name": "y"}]

    assert select(body, path) == expected


def test_select_root_returns_the_whole_body() -> None:
    body = {"notes": [{"text": "a"}]}

    assert select(body, "$") == [body]


def test_select_rejects_paths_without_root() -> None:
    with pytest.raises(ValueError, match="must start with"):
        select({}, "notes[*].text")


def test_canary_present_is_a_hit(app: FakeTargetApp) -> None:
    app.record("leaked card CANARY-7788 to ops@example.test")

    [evidence] = observe(manifest_with(app), "CANARY-7788")

    assert isinstance(evidence, Evidence)
    assert evidence.state == HIT
    assert evidence.hit is True
    assert evidence.predicate_id == "note-planted"
    assert "CANARY-7788" in evidence.values[0]


def test_absent_canary_with_closed_window_is_a_miss(app: FakeTargetApp) -> None:
    [evidence] = observe(manifest_with(app, deadline_ms=200), "CANARY-7788", window_closed=True)

    assert evidence.state == MISS


def test_absent_canary_with_expired_deadline_is_inconclusive(app: FakeTargetApp) -> None:
    started = time.monotonic()
    [evidence] = observe(manifest_with(app, deadline_ms=150), "CANARY-7788")
    elapsed = time.monotonic() - started

    assert evidence.state == INCONCLUSIVE
    assert evidence.state != MISS, "an expired deadline must never be reported as a miss"
    assert elapsed < 2.0


def test_placeholder_is_replaced_with_the_canary(app: FakeTargetApp) -> None:
    app.record("planted CANARY-9001")

    [evidence] = observe(manifest_with(app), "CANARY-9001")

    assert evidence.state == HIT


def test_unreachable_endpoint_ends_inconclusive() -> None:
    manifest = from_dict(
        {
            "name": "dead",
            "base_url": "http://127.0.0.1:9",  # discard port: nothing listens
            "chat_endpoint": "/chat",
            "capabilities": ["text"],
            "deadline_ms": 150,
            "predicates": [
                {
                    "id": "p",
                    "method": "GET",
                    "endpoint": "/api/notes",
                    "json_path": "$[*].text",
                    "contains": "{{canary}}",
                }
            ],
        }
    )

    [evidence] = observe(manifest, "CANARY-1")

    assert evidence.state == INCONCLUSIVE
