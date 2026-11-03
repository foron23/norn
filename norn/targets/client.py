"""HTTP/SSE client that drives a target application's chat endpoint.

The app is measured through the door its own users use: one ``POST`` to the chat
endpoint, answered with an incrementally flushed ``text/event-stream``. This
module turns that stream into a :class:`RunRecord` — the text the app produced,
the tool calls it attempted, the policy decisions that gated them, the retrieval
chunks it used — without touching files, probes or the app's database.

Transport is stdlib (``urllib``): no new dependency, and the stream is read line
by line as it arrives. A buffered stream is detected as a timeout, not smoothed
over.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from norn.targets.manifest import CANONICAL_EVENTS, TargetManifest
from norn.targets.sse import iter_events

#: Decisions that mean the app refused to perform the call.
BLOCKING_DECISIONS = frozenset({"deny", "approval_required"})


class TargetError(RuntimeError):
    """The target did not answer the way the contract says it must."""


@dataclass(frozen=True)
class ToolCall:
    """One ``tool.call`` joined with its ``tool.result`` and ``policy.decision``."""

    name: str
    args: dict[str, Any]
    turn: int
    result: str | None = None
    ok: bool | None = None
    decision: str | None = None
    rule: str | None = None

    @property
    def is_authorized(self) -> int:
        """1 unless the app's own policy blocked the call (legacy UAR input)."""
        return 0 if self.decision in BLOCKING_DECISIONS else 1

    def as_provider_dict(self) -> dict[str, Any]:
        """Shape the runtime persists (``tool_call_event`` / scoring context)."""
        return {
            "tool_name": self.name,
            "tool_params": json.dumps(self.args, sort_keys=True),
            "tool_result": self.result or "",
            "is_authorized": self.is_authorized,
            "turn": self.turn,
        }


@dataclass(frozen=True)
class RunRecord:
    """Everything one chat turn told us, plus what it failed to tell us."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    policy: tuple[dict[str, Any], ...] = ()
    retrieval: tuple[dict[str, Any], ...] = ()
    finished: bool = False
    timed_out: bool = False
    unknown_events: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    run_id: str | None = None
    session_id: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    @property
    def tool_call_dicts(self) -> list[dict[str, Any]]:
        """Provider-shaped tool calls, ready for persistence and scoring."""
        return [call.as_provider_dict() for call in self.tool_calls]


@dataclass
class _PendingCall:
    """Mutable accumulator for one ``tool.call`` while the stream is open."""

    name: str
    args: dict[str, Any]
    turn: int
    result: str | None = None
    ok: bool | None = None
    decision: str | None = None
    rule: str | None = None

    def freeze(self) -> ToolCall:
        return ToolCall(
            name=self.name,
            args=self.args,
            turn=self.turn,
            result=self.result,
            ok=self.ok,
            decision=self.decision,
            rule=self.rule,
        )


def build_event_index(manifest: TargetManifest) -> dict[str, str]:
    """Map the names the app emits to Norn's canonical names.

    Identity for the seven canonical events; ``manifest.event_map`` overrides
    only the entries the app names differently.
    """
    index = {name: name for name in CANONICAL_EVENTS}
    for canonical, app_name in manifest.event_map.items():
        index.pop(canonical, None)
        index[app_name] = canonical
    return index


def _url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _headers(manifest: TargetManifest, accept: str) -> dict[str, str]:
    headers = {"Accept": accept, "User-Agent": "norn/0.1.0"}
    token = os.environ.get(manifest.token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_lines(response: Any) -> Iterator[bytes]:
    """Yield raw lines as they arrive (never the whole body at once)."""
    while True:
        line = response.readline()
        if not line:
            return
        yield line


def _payload(event_name: str, raw: str, unknown: list[str]) -> dict[str, Any] | None:
    """Decode an event body; a non-object body is counted, never fatal."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        unknown.append(event_name)
        return None
    if not isinstance(data, dict):
        unknown.append(event_name)
        return None
    return data


def chat(
    manifest: TargetManifest,
    message: str,
    session_id: str | None = None,
    *,
    deadline_ms: int | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunRecord:
    """Drive one chat turn and return what the stream reported.

    Args:
        manifest: Validated target description.
        message: Payload to send as the user turn.
        session_id: Conversation to continue; a fresh one is generated if None.
        deadline_ms: Overrides ``manifest.deadline_ms`` (budget for the whole run).
        on_event: Optional callback fired per decoded event, in arrival order.

    Returns:
        A :class:`RunRecord`. ``finished`` is True only if ``run.finished``
        arrived; otherwise ``timed_out`` is True — a run that did not close is
        never reported as a completed run.

    Raises:
        TargetError: Non-2xx answer, or a stream that ends before any event.
    """
    budget_ms = manifest.deadline_ms if deadline_ms is None else deadline_ms
    sid = session_id or uuid.uuid4().hex
    url = _url(manifest.base_url, manifest.chat_endpoint)
    body = json.dumps({"message": message, "session_id": sid}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            **_headers(manifest, "text/event-stream"),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    index = build_event_index(manifest)

    text: list[str] = []
    calls: list[_PendingCall] = []
    policy: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    unknown: list[str] = []
    run_id: str | None = None
    finished = False
    timed_out = False
    tokens_in = tokens_out = 0
    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=budget_ms / 1000) as response:
            if response.status >= 400:
                raise TargetError(f"chat endpoint answered HTTP {response.status}")
            for event in iter_events(_read_lines(response)):
                if event.event == "message" and not event.data:
                    continue  # bare keepalive block
                canonical = index.get(event.event)
                if canonical is None:
                    unknown.append(event.event)
                    continue
                data = _payload(event.event, event.data, unknown)
                if data is None:
                    continue
                if on_event is not None:
                    on_event(canonical, data)

                if canonical == "run.started":
                    run_id = str(data.get("run_id") or run_id or "")
                    sid = str(data.get("session_id") or sid)
                elif canonical == "message.delta":
                    text.append(str(data.get("text", "")))
                elif canonical == "retrieval.chunks":
                    retrieval.append(
                        {
                            "query": data.get("query", ""),
                            "chunks": data.get("chunks", []),
                        }
                    )
                elif canonical == "tool.call":
                    calls.append(
                        _PendingCall(
                            name=str(data.get("name", "")),
                            args=data.get("args") or {},
                            turn=int(data.get("turn", len(calls) + 1)),
                        )
                    )
                elif canonical == "tool.result":
                    _attach_result(calls, data)
                elif canonical == "policy.decision":
                    _attach_decision(calls, data)
                    policy.append(data)
                elif canonical == "run.finished":
                    finished = True
                    run_id = str(data.get("run_id") or run_id or "")
                    usage = data.get("usage") or {}
                    tokens_in = int(usage.get("tokens_in", 0) or 0)
                    tokens_out = int(usage.get("tokens_out", 0) or 0)

                elapsed_ms = (time.monotonic() - started) * 1000
                if elapsed_ms > budget_ms and not finished:
                    timed_out = True
                    break
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", errors="replace")
        raise TargetError(f"chat endpoint answered HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TargetError(f"cannot reach target at {url}: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        # Socket read timed out: the deadline expired mid-stream.
        if finished:
            raise TargetError(f"stream broke after run.finished: {exc}") from exc
        timed_out = True

    latency_ms = round((time.monotonic() - started) * 1000, 1)
    frozen = [call.freeze() for call in calls]
    return RunRecord(
        text="".join(text),
        tool_calls=tuple(frozen),
        policy=tuple(policy),
        retrieval=tuple(retrieval),
        finished=finished,
        timed_out=timed_out or not finished,
        unknown_events=tuple(unknown),
        violations=_violations(frozen, finished),
        run_id=run_id,
        session_id=sid,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


def _attach_result(calls: list[_PendingCall], data: dict[str, Any]) -> None:
    """Join a ``tool.result`` to the most recent call with the same name."""
    name = str(data.get("name", ""))
    result = data.get("result")
    for call in reversed(calls):
        if call.name == name and call.result is None:
            call.result = result if isinstance(result, str) else json.dumps(result)
            call.ok = data.get("ok")
            return


def _attach_decision(calls: list[_PendingCall], data: dict[str, Any]) -> None:
    """Join a ``policy.decision`` to the most recent unresolved call."""
    tool = str(data.get("tool", ""))
    for call in reversed(calls):
        if call.name == tool and call.decision is None:
            call.decision = str(data.get("decision", "")) or None
            call.rule = data.get("rule")
            return


def _violations(calls: list[ToolCall], finished: bool) -> tuple[str, ...]:
    """Report breaks of the stream invariants (they make the verdict invalid)."""
    problems: list[str] = []
    for call in calls:
        if call.result is None:
            problems.append(f"tool.call without tool.result: {call.name}")
        if call.decision is None:
            problems.append(f"tool.call without policy.decision: {call.name}")
    if not finished:
        problems.append("stream closed without run.finished")
    return tuple(problems)


class TargetClient:
    """Driver bound to one target manifest.

    ``run_campaign`` uses this when ``CampaignConfig.target_manifest`` is set:
    one replica is one app run (one ``POST`` to the chat endpoint), so the
    per-provider turn loop does not apply.
    """

    def __init__(self, manifest: TargetManifest) -> None:
        self.manifest = manifest

    def chat(self, message: str, session_id: str | None = None, **kwargs: Any) -> RunRecord:
        """Drive one app run for ``message``."""
        return chat(self.manifest, message, session_id, **kwargs)

    @property
    def capabilities(self) -> frozenset[str]:
        """What this target can report, per its manifest."""
        return self.manifest.capabilities
