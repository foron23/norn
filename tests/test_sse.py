"""Tests for the stdlib-only SSE parser."""
from __future__ import annotations

from collections.abc import Iterator

from norn.targets.sse import SSEEvent, iter_events


def test_simple_event_with_explicit_type_and_id() -> None:
    stream = ["event: token\n", "id: 7\n", "data: hello\n", "\n"]

    events = list(iter_events(stream))

    assert events == [SSEEvent(event="token", data="hello", id="7")]


def test_event_without_name_defaults_to_message() -> None:
    events = list(iter_events(["data: plain\n\n"]))

    assert events == [SSEEvent(event="message", data="plain", id=None)]


def test_multiple_data_lines_are_joined_with_newlines() -> None:
    stream = ["data: first\n", "data: second\n", "data: third\n", "\n"]

    events = list(iter_events(stream))

    assert len(events) == 1
    assert events[0].data == "first\nsecond\nthird"


def test_keepalive_comments_produce_no_events() -> None:
    stream = [":keepalive\n", "\n", "data: real\n", "\n", ":keepalive\n"]

    events = list(iter_events(stream))

    assert [event.data for event in events] == ["real"]


def test_crlf_line_endings_are_accepted() -> None:
    stream = [b"event: token\r\n", b"data: hi\r\n", b"\r\n"]

    events = list(iter_events(stream))

    assert events == [SSEEvent(event="token", data="hi", id=None)]


def test_eof_dispatches_trailing_block_without_blank_line() -> None:
    stream = ["data: tail\n"]

    events = list(iter_events(stream))

    assert [event.data for event in events] == ["tail"]


def test_done_sentinel_ends_the_stream() -> None:
    stream = ["data: before\n\n", "data: [DONE]\n\n", "data: after\n\n"]

    events = list(iter_events(stream))

    assert [event.data for event in events] == ["before"]


def test_events_are_yielded_before_the_next_block_is_read() -> None:
    pulled: list[int] = []

    def stream() -> Iterator[str]:
        pulled.append(1)
        yield "data: first\n\n"
        pulled.append(2)
        yield "data: second\n\n"

    events = iter_events(stream())

    assert next(events).data == "first"
    assert pulled == [1], "parser read ahead instead of yielding incrementally"
    assert next(events).data == "second"
