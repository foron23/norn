"""Shared retry/backoff helpers for provider HTTP clients (NOR-05).

Both ``OpenAICompatibleClient`` and ``OllamaClient`` retry transient HTTP
failures (rate limits and 5xx server errors) with exponential backoff,
jitter, and support for the ``Retry-After`` response header.
"""
from __future__ import annotations

import random
import time

import httpx

# HTTP status codes safe to retry: rate limit + transient server errors.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Exponential backoff delays per attempt (0-based): 1s, 2s, 4s.
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

DEFAULT_RETRY_ATTEMPTS = 3


def backoff_delay(attempt: int, *, jitter_ratio: float = 0.1) -> float:
    """Return the sleep delay for retry ``attempt`` (0-based), with jitter."""
    base = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
    return base + random.uniform(0.0, jitter_ratio * base)


def parse_retry_after(resp: httpx.Response, default: float) -> float:
    """Respect the ``Retry-After`` header when present, else ``default``."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return default
    try:
        return max(default, float(raw))
    except ValueError:
        return default


def with_retry(request_fn, *, attempts: int = DEFAULT_RETRY_ATTEMPTS) -> httpx.Response:
    """Execute ``request_fn`` up to ``attempts`` times with backoff.

    ``request_fn`` must return an :class:`httpx.Response`. Responses with a
    retryable status code (429, 500, 502, 503, 504) are retried with
    exponential backoff + jitter (respecting ``Retry-After`` when present).
    The last response is returned once attempts are exhausted so callers can
    raise the appropriate domain error.
    """
    last_resp: httpx.Response | None = None
    for attempt in range(attempts):
        last_resp = request_fn()
        if last_resp.status_code not in RETRYABLE_STATUS_CODES or attempt >= attempts - 1:
            return last_resp
        time.sleep(parse_retry_after(last_resp, backoff_delay(attempt)))
    assert last_resp is not None
    return last_resp
