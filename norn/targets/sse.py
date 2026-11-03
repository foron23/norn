"""Minimal Server-Sent Events parser (stdlib only, transport level).

This module knows the wire format and nothing else: no HTTP, no JSON
interpretation, no knowledge of the seven events in the target contract. The
client feeds it the lines coming off the socket and gets one :class:`SSEEvent`
per dispatched block, in the order they arrived.

Incrementality is part of the contract: :func:`iter_events` is a generator that
yields a block as soon as its terminating blank line shows up. It never buffers
the whole stream first — a buffered stream is not a stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

#: Sentinel the app may send as the last body to close the stream.
DONE_SENTINEL = "[DONE]"

#: Event name used when a block carries `data:` but no `event:` field.
DEFAULT_EVENT = "message"


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched SSE block."""

    event: str
    data: str
    id: str | None = None


def _decode(chunk: str | bytes) -> str:
    """Return a chunk as text (urllib hands us bytes)."""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return chunk


def _iter_lines(chunks: Iterable[str | bytes]) -> Iterator[str]:
    """Yield complete lines, whatever chunk boundaries the transport used.

    A trailing partial line is carried over to the next chunk, so callers can
    hand over arbitrary network reads (one line, several lines, half a line).
    """
    carry = ""
    for chunk in chunks:
        text = carry + _decode(chunk)
        carry = ""
        parts = text.split("\n")
        if parts[-1]:
            carry = parts.pop()  # incomplete line: keep it for the next chunk
        else:
            parts.pop()  # chunk ended with a newline: nothing pending
        for line in parts:
            yield line.rstrip("\r")
    if carry:
        yield carry.rstrip("\r")


def _field_value(line: str) -> tuple[str, str]:
    """Split ``field: value`` and strip the single optional leading space."""
    name, _, value = line.partition(":")
    return name, value.removeprefix(" ")


def iter_events(chunks: Iterable[str | bytes]) -> Iterator[SSEEvent]:
    """Yield one :class:`SSEEvent` per blank-line-terminated block.

    Handles ``\\n`` and ``\\r\\n`` endings, ``:comment`` keepalives (ignored),
    several ``data:`` lines in one block (joined with ``\\n``), a missing
    ``event:`` field (defaults to ``message``), a trailing block without a final
    blank line, and the ``data: [DONE]`` sentinel, which ends the iteration.
    """
    event_name = DEFAULT_EVENT
    event_id: str | None = None
    data: list[str] = []

    for line in _iter_lines(chunks):
        if not line:
            if data:
                payload = "\n".join(data)
                if payload == DONE_SENTINEL:
                    return
                yield SSEEvent(event=event_name, data=payload, id=event_id)
            event_name = DEFAULT_EVENT
            event_id = None
            data = []
            continue

        if line.startswith(":"):
            continue  # comment / keepalive

        name, value = _field_value(line)
        if name == "event":
            event_name = value
        elif name == "data":
            data.append(value)
        elif name == "id":
            event_id = value
        # Unknown fields are ignored, as the SSE spec requires.

    if data:  # stream ended without a trailing blank line
        payload = "\n".join(data)
        if payload != DONE_SENTINEL:
            yield SSEEvent(event=event_name, data=payload, id=event_id)
