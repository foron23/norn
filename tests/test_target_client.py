"""Tests for the HTTP/SSE chat client against the fake streaming target."""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self

import pytest

from norn.targets.client import RunRecord, TargetError, build_event_index, chat
from norn.targets.manifest import from_dict
from tests.fixtures.fake_target_app import FakeTargetApp


def manifest_for(app: FakeTargetApp, endpoint: str = "/attack", **overrides):
    data = {
        "name": "fake",
        "base_url": app.base_url,
        "chat_endpoint": endpoint,
        "capabilities": ["text", "stream", "tool_calls", "policy", "retrieval"],
        "deadline_ms": 4000,
    }
    data.update(overrides)
    return from_dict(data)


# ── Stub server for event shapes the fake app does not emit ──────────────────

class _StubHandler(BaseHTTPRequestHandler):
    """Streams ``app.events`` with an optional delay, then closes."""

    def do_POST(self) -> None:
        events: Sequence[tuple[str, str]] = self.server.events  # type: ignore[attr-defined]
        delay: float = self.server.delay  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for name, body in events:
            self.wfile.write(f"event: {name}\ndata: {body}\n\n".encode())
            self.wfile.flush()
            if delay:
                time.sleep(delay)
        self.close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:
        return


class StubTarget:
    """Minimal streaming server with caller-defined events."""

    def __init__(self, events: Sequence[tuple[str, str]], delay: float = 0.0) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self._httpd.events = events  # type: ignore[attr-defined]
        self._httpd.delay = delay  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"


@pytest.fixture
def app() -> Iterator[FakeTargetApp]:
    with FakeTargetApp() as fake:
        yield fake


# ── Client behaviour ────────────────────────────────────────────────────────

def test_attack_run_is_fully_recorded(app: FakeTargetApp) -> None:
    record = chat(manifest_for(app), "hello canary")

    assert isinstance(record, RunRecord)
    assert record.finished is True
    assert record.timed_out is False
    assert record.text == "Processed: hello canary"
    assert len(record.tool_calls) == 1
    call = record.tool_calls[0]
    assert call.name == "send"
    assert call.args == {"to": "ops@example.test"}
    assert call.turn == 1
    assert call.result == "queued"
    assert call.ok is True
    assert call.decision == "allow"
    assert call.rule == "default-allow"
    assert call.is_authorized == 1
    assert len(record.retrieval) == 1
    assert record.retrieval[0]["chunks"][0]["id"] == "kb-001"
    assert (record.tokens_in, record.tokens_out) == (128, 42)
    assert record.unknown_events == ()
    assert record.violations == ()
    assert record.run_id
    assert record.session_id
    assert record.latency_ms >= 0


def test_tool_calls_are_provider_shaped(app: FakeTargetApp) -> None:
    record = chat(manifest_for(app), "payload")

    [call] = record.tool_call_dicts
    assert set(call) == {"tool_name", "tool_params", "tool_result", "is_authorized", "turn"}
    assert call["tool_name"] == "send"
    assert json.loads(call["tool_params"]) == {"to": "ops@example.test"}
    assert call["is_authorized"] == 1


def test_session_id_is_reused_when_given(app: FakeTargetApp) -> None:
    record = chat(manifest_for(app), "hi", "sess-42")

    assert record.session_id == "sess-42"


def test_benign_run_has_no_tool_traffic(app: FakeTargetApp) -> None:
    record = chat(manifest_for(app, "/benign"), "how do I reset my password?")

    assert record.finished is True
    assert record.timed_out is False
    assert record.tool_calls == ()
    assert record.violations == ()


def test_stream_without_run_finished_is_timed_out(app: FakeTargetApp) -> None:
    record = chat(manifest_for(app, "/truncated"), "payload")

    assert record.finished is False
    assert record.timed_out is True
    assert "stream closed without run.finished" in record.violations
    assert len(record.tool_calls) == 1  # the calls before the cut are still there


def test_blocking_decision_marks_the_call_unauthorized() -> None:
    events = [
        ("tool.call", '{"name": "send", "args": {}, "turn": 1}'),
        ("tool.result", '{"name": "send", "ok": false, "result": "denied"}'),
        ("policy.decision", '{"tool": "send", "decision": "deny", "rule": "no-bec"}'),
        ("run.finished", '{"run_id": "r1", "usage": {"tokens_in": 1, "tokens_out": 2}}'),
    ]
    with StubTarget(events) as stub:
        manifest = from_dict(
            {
                "name": "stub",
                "base_url": stub.base_url,
                "chat_endpoint": "/chat",
                "capabilities": ["text", "stream", "tool_calls", "policy"],
            }
        )
        record = chat(manifest, "payload")

    assert record.finished is True
    assert record.tool_calls[0].decision == "deny"
    assert record.tool_calls[0].is_authorized == 0


def test_event_map_translates_app_specific_names() -> None:
    events = [
        ("start", '{"run_id": "r9", "session_id": "s9"}'),
        ("token", '{"text": "hey"}'),
        ("token", '{"text": " there"}'),
        ("done", '{"run_id": "r9", "usage": {"tokens_in": 3, "tokens_out": 4}}'),
    ]
    with StubTarget(events) as stub:
        manifest = from_dict(
            {
                "name": "mapped",
                "base_url": stub.base_url,
                "chat_endpoint": "/chat",
                "capabilities": ["text", "stream"],
                "event_map": {
                    "run.started": "start",
                    "message.delta": "token",
                    "run.finished": "done",
                },
            }
        )
        record = chat(manifest, "hello")

    assert record.finished is True
    assert record.text == "hey there"
    assert record.run_id == "r9"
    assert (record.tokens_in, record.tokens_out) == (3, 4)
    assert record.unknown_events == ()


def test_unknown_and_malformed_events_are_counted_not_fatal() -> None:
    events = [
        ("run.started", '{"run_id": "r1", "session_id": "s1"}'),
        ("weird.event", '{"whatever": true}'),
        ("message.delta", "not-json-at-all"),
        ("message.delta", '{"text": "still here"}'),
        ("run.finished", '{"run_id": "r1", "usage": {"tokens_in": 1, "tokens_out": 1}}'),
    ]
    with StubTarget(events) as stub:
        manifest = from_dict(
            {
                "name": "noisy",
                "base_url": stub.base_url,
                "chat_endpoint": "/chat",
                "capabilities": ["text", "stream"],
            }
        )
        record = chat(manifest, "payload")

    assert record.finished is True
    assert record.text == "still here"
    assert record.unknown_events == ("weird.event", "message.delta")


def test_unreachable_endpoint_is_a_target_error(app: FakeTargetApp) -> None:
    with pytest.raises(TargetError, match="HTTP 404"):
        chat(manifest_for(app, "/does-not-exist"), "payload")


def test_events_arrive_incrementally() -> None:
    events = [
        ("message.delta", '{"text": "first"}'),
        ("message.delta", '{"text": "second"}'),
        ("run.finished", '{"run_id": "r1", "usage": {"tokens_in": 0, "tokens_out": 0}}'),
    ]
    seen: list[tuple[str, float]] = []
    with StubTarget(events, delay=0.2) as stub:
        manifest = from_dict(
            {
                "name": "slow",
                "base_url": stub.base_url,
                "chat_endpoint": "/chat",
                "capabilities": ["text", "stream"],
            }
        )
        started = time.monotonic()
        record = chat(
            manifest,
            "payload",
            on_event=lambda name, data: seen.append((name, time.monotonic() - started)),
        )

    assert record.finished is True
    assert [name for name, _ in seen] == ["message.delta", "message.delta", "run.finished"]
    first_seen = seen[0][1]
    finished_seen = seen[-1][1]
    assert first_seen < finished_seen - 0.25, "the client waited for the whole body"


def test_event_index_defaults_to_identity() -> None:
    manifest = from_dict(
        {
            "name": "x",
            "base_url": "http://127.0.0.1:1",
            "chat_endpoint": "/chat",
            "capabilities": ["text"],
        }
    )

    index = build_event_index(manifest)

    assert index["run.finished"] == "run.finished"
    assert index["policy.decision"] == "policy.decision"
