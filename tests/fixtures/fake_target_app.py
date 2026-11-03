"""Stdlib-only fake target application for Norn's HTTP/SSE client tests.

This is a test double, not the lab app: no UI, no persistence, no deploy. It
exists so the generic HTTP/SSE target client and its preflight gate can be
developed and tested against a deterministic server that speaks the same
transport shape as a real streaming chat app.

Transport contract
------------------
* Server: ``ThreadingHTTPServer`` bound to ``127.0.0.1`` on port 0 by default,
  so parallel tests never collide. ``FakeTargetApp.base_url`` reports the
  actual bound address.
* SSE responses advertise ``Content-Type: text/event-stream``, send **no**
  ``Content-Length`` (the body is delimited by connection close) and flush
  after *every* event, so a client can observe events as they happen.
* Chat routes keep the connection open until the stream ends, then close it.
* Auth is NOT enforced: requests are accepted with or without an
  ``Authorization`` header. The header is read and ignored (it never raises),
  which keeps the double usable for both authenticated-looking and bare
  requests. Note that this means the double cannot be used to test auth
  handling beyond "the client sends a header without breaking".

Routes
------
``POST /attack`` (SSE) emits, in order::

    run.started -> message.delta -> retrieval.chunks -> tool.call
    -> tool.result -> policy.decision -> run.finished

``POST /benign`` (SSE) emits::

    run.started -> message.delta -> run.finished

``POST /truncated`` (SSE) emits the first six ``/attack`` events and closes
without ``run.finished``, so a client must mark the run as timed out.

``GET /api/crm/notes`` and ``GET /api/outbox`` read back the messages recorded
by ``/attack`` (and ``/truncated``), which is how a planted canary becomes
observable evidence downstream.

Standalone use::

    python tests/fixtures/fake_target_app.py --port 8099

serves the same routes on ``http://127.0.0.1:8099`` for preflight checks.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

Event = tuple[str, dict[str, Any]]

ATTACK_EVENT_ORDER: tuple[str, ...] = (
    "run.started",
    "message.delta",
    "retrieval.chunks",
    "tool.call",
    "tool.result",
    "policy.decision",
    "run.finished",
)

BENIGN_EVENT_ORDER: tuple[str, ...] = (
    "run.started",
    "message.delta",
    "run.finished",
)

# /truncated drops exactly the last event of /attack (run.finished).
TRUNCATED_EVENT_COUNT = len(ATTACK_EVENT_ORDER) - 1


def attack_events(
    run_id: str, session_id: str, message: str
) -> list[Event]:
    """Build the full ``/attack`` event sequence for one request."""
    return [
        ("run.started", {"run_id": run_id, "session_id": session_id}),
        ("message.delta", {"text": f"Processed: {message}"}),
        (
            "retrieval.chunks",
            {
                "query": message,
                "chunks": [
                    {"id": "kb-001", "score": 0.93},
                    {"id": "kb-002", "score": 0.71},
                ],
            },
        ),
        (
            "tool.call",
            {"name": "send", "args": {"to": "ops@example.test"}, "turn": 1},
        ),
        ("tool.result", {"name": "send", "ok": True, "result": "queued"}),
        (
            "policy.decision",
            {"tool": "send", "decision": "allow", "rule": "default-allow"},
        ),
        (
            "run.finished",
            {
                "run_id": run_id,
                "usage": {"tokens_in": 128, "tokens_out": 42},
            },
        ),
    ]


def benign_events(run_id: str, session_id: str, message: str) -> list[Event]:
    """Build the ``/benign`` event sequence: no tool traffic, no storage."""
    return [
        ("run.started", {"run_id": run_id, "session_id": session_id}),
        ("message.delta", {"text": f"Echo: {message}"}),
        (
            "run.finished",
            {
                "run_id": run_id,
                "usage": {"tokens_in": 24, "tokens_out": 8},
            },
        ),
    ]


class FakeTargetApp:
    """In-process fake streaming target, safe to run one per test.

    ``event_delay`` seconds are slept after each SSE event write; tests raise
    it to prove events are observed before the stream completes.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        event_delay: float = 0.0,
    ) -> None:
        self.host = host
        self.port = port
        self.event_delay = event_delay
        self._store_lock = threading.Lock()
        self._notes: list[str] = []
        self._outbox: list[str] = []
        self._httpd: _FakeTargetHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------
    def start(self) -> FakeTargetApp:
        """Bind and serve in a background daemon thread; return ``self``."""
        httpd = _FakeTargetHTTPServer((self.host, self.port), _FakeTargetHandler)
        httpd.app = self
        self._httpd = httpd
        self.port = httpd.server_address[1]
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="fake-target-app",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the socket. Idempotent."""
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread = None, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5.0)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        """Base URL including the bound port, e.g. ``http://127.0.0.1:54321``."""
        return f"http://{self.host}:{self.port}"

    # -- store ---------------------------------------------------------
    def record(self, message: str) -> None:
        """Record a message so the GET endpoints can serve it back."""
        with self._store_lock:
            self._notes.append(message)
            self._outbox.append(message)

    def notes(self) -> list[dict[str, Any]]:
        """Snapshot of stored notes, one row per recorded message."""
        with self._store_lock:
            messages = list(self._notes)
        return [{"id": i, "text": text} for i, text in enumerate(messages, 1)]

    def outbox(self) -> list[dict[str, Any]]:
        """Snapshot of the outbox produced by recorded messages."""
        with self._store_lock:
            messages = list(self._outbox)
        return [
            {"id": i, "to": "ops@example.test", "body": text}
            for i, text in enumerate(messages, 1)
        ]

    # -- request handling (called from the handler thread) -------------
    def chat_events(
        self, path: str, message: str, session_id: str
    ) -> tuple[list[Event], str | None]:
        """Return ``(events, recorded_message)`` for a chat route."""
        run_id = uuid.uuid4().hex
        if path == "/attack":
            events = attack_events(run_id, session_id, message)
            return events, message
        if path == "/truncated":
            events = attack_events(run_id, session_id, message)
            return events[:TRUNCATED_EVENT_COUNT], message
        if path == "/benign":
            return benign_events(run_id, session_id, message), None
        raise KeyError(path)


class _FakeTargetHTTPServer(ThreadingHTTPServer):
    """Threading server carrying a back-reference to the fake app."""

    daemon_threads = True
    allow_reuse_address = True
    app: FakeTargetApp


class _FakeTargetHandler(BaseHTTPRequestHandler):
    """Route HTTP requests to the fake app's fixed contract."""

    protocol_version = "HTTP/1.1"
    server_version = "FakeTargetApp/1.0"
    sys_version = ""

    @property
    def app(self) -> FakeTargetApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence per-request logging so test output stays readable."""

    # -- verbs ---------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/api/crm/notes":
            self._send_json(200, {"notes": self.app.notes()})
        elif self.path == "/api/outbox":
            self._send_json(200, {"items": self.app.outbox()})
        else:
            self._send_not_found()

    def do_POST(self) -> None:
        if self.path not in ("/attack", "/benign", "/truncated"):
            self._send_not_found()
            return
        message, session_id = self._read_chat_body()
        events, recorded = self.app.chat_events(
            self.path, message, session_id
        )
        if recorded is not None:
            self.app.record(recorded)
        self._send_sse(events)

    # -- helpers -------------------------------------------------------
    def _read_chat_body(self) -> tuple[str, str]:
        """Parse the JSON chat body, tolerating missing fields."""
        raw = b""
        length = self.headers.get("Content-Length")
        if length:
            try:
                raw = self.rfile.read(int(length))
            except (ValueError, OSError):
                raw = b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        message = payload.get("message")
        session_id = payload.get("session_id")
        # The Authorization header is deliberately read-and-ignored: the
        # double accepts both authenticated-looking and bare requests.
        _ = self.headers.get("Authorization")
        return (
            message if isinstance(message, str) else "",
            session_id if isinstance(session_id, str) else uuid.uuid4().hex,
        )

    def _send_sse(self, events: Sequence[Event]) -> None:
        """Stream events one at a time, flushed, without Content-Length."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for name, data in events:
                chunk = f"event: {name}\ndata: {json.dumps(data)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
                if self.app.event_delay:
                    time.sleep(self.app.event_delay)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def _send_not_found(self) -> None:
        self._send_json(404, {"error": "not found"})


def serve(port: int = 8099, host: str = "127.0.0.1") -> None:
    """Serve the fake target in the foreground until interrupted."""
    app = FakeTargetApp(host=host, port=port)
    app.start()
    print(f"fake target app listening on {app.base_url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        app.stop()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python tests/fixtures/fake_target_app.py``."""
    parser = argparse.ArgumentParser(
        description="Serve the stdlib fake target app used by Norn tests."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="TCP port to bind (default: 8099)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args(argv)
    serve(port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
