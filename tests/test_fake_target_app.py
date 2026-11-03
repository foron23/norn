"""Smoke tests for the stdlib fake target app test double.

The streams are read with plain ``urllib`` on purpose: the generic SSE client
is exactly what this double exists to support, so the double cannot depend on
it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from tests.fixtures.fake_target_app import (
    ATTACK_EVENT_ORDER,
    BENIGN_EVENT_ORDER,
    FakeTargetApp,
)

CHAT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer not-checked-by-the-double",
}


@pytest.fixture()
def app() -> Iterator[FakeTargetApp]:
    """Started fake target on an ephemeral port, stopped at teardown."""
    instance = FakeTargetApp()
    instance.start()
    yield instance
    instance.stop()


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    """Split an SSE byte stream into ``(event_name, data)`` pairs."""
    events: list[tuple[str, dict]] = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        assert name is not None, f"SSE block without event name: {block!r}"
        events.append((name, json.loads("\n".join(data_lines))))
    return events


def post_chat(url: str, message: str, session_id: str = "sess-1"):
    """Open a streaming POST and return the live response object."""
    body = json.dumps({"message": message, "session_id": session_id})
    request = urllib.request.Request(
        url, data=body.encode("utf-8"), headers=CHAT_HEADERS, method="POST"
    )
    return urllib.request.urlopen(request, timeout=10)


def collect_chat(url: str, message: str) -> list[tuple[str, dict]]:
    """Consume a whole chat stream and parse it."""
    with post_chat(url, message) as response:
        return parse_sse(response.read())


def test_attack_stream_is_incremental_and_ordered(app: FakeTargetApp) -> None:
    """Events are visible before the stream ends, in the documented order."""
    app.event_delay = 0.15
    started_at = time.monotonic()
    response = post_chat(app.base_url + "/attack", "planted-canary")
    assert response.headers["Content-Type"].startswith("text/event-stream")
    assert response.headers.get("Content-Length") is None

    first_chunk = response.read1(4096)
    first_event_at = time.monotonic() - started_at
    rest = response.read()
    finished_at = time.monotonic() - started_at
    response.close()

    assert b"event: run.started" in first_chunk
    assert b"run.finished" not in first_chunk
    assert first_event_at < finished_at - 0.3

    events = parse_sse(first_chunk + rest)
    assert [name for name, _ in events] == list(ATTACK_EVENT_ORDER)

    by_name = dict(events)
    assert by_name["run.started"]["session_id"] == "sess-1"
    assert by_name["run.started"]["run_id"] == by_name["run.finished"]["run_id"]
    assert by_name["tool.call"]["name"] == "send"
    assert by_name["tool.call"]["args"] == {"to": "ops@example.test"}
    assert by_name["tool.result"]["ok"] is True
    assert by_name["policy.decision"]["decision"] == "allow"
    assert set(by_name["run.finished"]["usage"]) == {"tokens_in", "tokens_out"}


def test_benign_stream_has_no_tool_events(app: FakeTargetApp) -> None:
    """The benign route streams text and finishes without tool or policy events."""
    events = collect_chat(app.base_url + "/benign", "hello")

    assert [name for name, _ in events] == list(BENIGN_EVENT_ORDER)
    assert not any(name.startswith("tool.") for name, _ in events)
    assert "policy.decision" not in dict(events)


def test_truncated_stream_ends_without_run_finished(app: FakeTargetApp) -> None:
    """The truncated route stops after six events so clients mark a timeout."""
    events = collect_chat(app.base_url + "/truncated", "cut me off")

    names = [name for name, _ in events]
    assert len(names) == 6
    assert names == list(ATTACK_EVENT_ORDER[:-1])
    assert "run.finished" not in names


def test_get_endpoints_return_the_stored_message(app: FakeTargetApp) -> None:
    """An attack run's message becomes readable evidence on both GET routes."""
    message = "planted-canary-7f3a"
    collect_chat(app.base_url + "/attack", message)

    with urllib.request.urlopen(
        app.base_url + "/api/crm/notes", timeout=5
    ) as response:
        notes = json.loads(response.read())
    with urllib.request.urlopen(
        app.base_url + "/api/outbox", timeout=5
    ) as response:
        outbox = json.loads(response.read())

    assert notes == {"notes": [{"id": 1, "text": message}]}
    assert outbox == {
        "items": [
            {"id": 1, "to": "ops@example.test", "body": message}
        ]
    }


def test_unknown_route_returns_404_json(app: FakeTargetApp) -> None:
    """Anything outside the contract answers 404 with a JSON error body."""
    with pytest.raises(urllib.error.HTTPError) as get_error:
        urllib.request.urlopen(app.base_url + "/api/nope", timeout=5)
    assert get_error.value.code == 404
    assert json.loads(get_error.value.read()) == {"error": "not found"}

    request = urllib.request.Request(
        app.base_url + "/api/nope",
        data=b"{}",
        headers=CHAT_HEADERS,
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as post_error:
        urllib.request.urlopen(request, timeout=5)
    assert post_error.value.code == 404
    assert json.loads(post_error.value.read()) == {"error": "not found"}
