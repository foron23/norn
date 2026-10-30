"""Unit tests for shared retry/backoff helpers (NOR-05 review fixes).

Covers the two review findings on ``norn/runtime/retry.py``:
- ``parse_retry_after`` must handle the HTTP-date format (RFC 9110).
- ``with_retry`` must reject invalid ``attempts`` values with a clear error.
"""
from __future__ import annotations

import datetime
from email.utils import format_datetime

import httpx
import pytest

from norn.runtime.retry import parse_retry_after, with_retry

# ── parse_retry_after ────────────────────────────────────────────────────────

def test_parse_retry_after_numeric():
    """Numeric Retry-After is parsed and wins over the default."""
    resp = httpx.Response(429, headers={"Retry-After": "3"})
    assert parse_retry_after(resp, 1.0) == 3.0


def test_parse_retry_after_numeric_lower_than_default():
    """Numeric Retry-After smaller than default never shortens the delay."""
    resp = httpx.Response(429, headers={"Retry-After": "0.5"})
    assert parse_retry_after(resp, 1.0) == 1.0


def test_parse_retry_after_http_date():
    """HTTP-date Retry-After (RFC 9110) is converted to a delay in seconds."""
    future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=60)
    resp = httpx.Response(429, headers={"Retry-After": format_datetime(future, usegmt=True)})
    delay = parse_retry_after(resp, 1.0)
    # Wide band: proves the date was parsed (delay >> default) without
    # depending on tight timing between formatting and parsing.
    assert 30.0 <= delay <= 61.0


def test_parse_retry_after_past_date_falls_back():
    """A Retry-After date in the past falls back to the default delay."""
    past = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=60)
    resp = httpx.Response(429, headers={"Retry-After": format_datetime(past, usegmt=True)})
    assert parse_retry_after(resp, 2.5) == 2.5


def test_parse_retry_after_invalid_falls_back():
    """An unparseable Retry-After falls back to the default delay."""
    resp = httpx.Response(429, headers={"Retry-After": "not-a-date"})
    assert parse_retry_after(resp, 2.5) == 2.5


def test_parse_retry_after_nan_inf_falls_back():
    """NaN/inf/negative Retry-After values fall back (never reach time.sleep)."""
    for bad in ("nan", "inf", "-inf", "-5"):
        resp = httpx.Response(429, headers={"Retry-After": bad})
        assert parse_retry_after(resp, 2.5) == 2.5


def test_parse_retry_after_missing_returns_default():
    """No Retry-After header returns the default delay."""
    resp = httpx.Response(429)
    assert parse_retry_after(resp, 2.5) == 2.5


# ── with_retry ───────────────────────────────────────────────────────────────

def test_with_retry_rejects_zero_attempts():
    """attempts=0 raises a clear ValueError instead of an assertion."""
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        with_retry(lambda: httpx.Response(200), attempts=0)


def test_with_retry_rejects_negative_attempts():
    """attempts<0 raises a clear ValueError."""
    with pytest.raises(ValueError, match="attempts must be >= 1"):
        with_retry(lambda: httpx.Response(200), attempts=-2)


def test_client_constructors_reject_invalid_retry_attempts():
    """Client constructors fail fast with a clear error for retry_attempts<1."""
    from norn.runtime.ollama_client import OllamaClient
    from norn.runtime.openai_client import OpenAICompatibleClient

    with pytest.raises(ValueError, match="retry_attempts must be >= 1"):
        OpenAICompatibleClient(retry_attempts=0)
    with pytest.raises(ValueError, match="retry_attempts must be >= 1"):
        OllamaClient(retry_attempts=0)


def test_with_retry_retries_403_throttling(monkeypatch):
    """A relay 403 is throttling, not auth: it must be retried (NOR-31)."""
    import norn.runtime.retry as retry_mod

    sleeps = []
    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def request_fn():
        calls["n"] += 1
        return httpx.Response(200 if calls["n"] == 2 else 403)

    resp = retry_mod.with_retry(request_fn, attempts=3)
    assert resp.status_code == 200
    assert calls["n"] == 2
    assert sleeps and sleeps[0] >= retry_mod.THROTTLE_MIN_DELAY_SECONDS


def test_with_retry_does_not_retry_401(monkeypatch):
    """A 401 is a credential problem and stays non-retryable."""
    import norn.runtime.retry as retry_mod

    sleeps = []
    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def request_fn():
        calls["n"] += 1
        return httpx.Response(401)

    resp = retry_mod.with_retry(request_fn, attempts=3)
    assert resp.status_code == 401
    assert calls["n"] == 1
    assert sleeps == []
