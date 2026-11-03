"""Tests for the target preflight (negative + positive evidence legs)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from norn.targets import preflight as preflight_module
from norn.targets.client import TargetError
from norn.targets.manifest import from_dict
from norn.targets.preflight import (
    OK,
    ORACLE_BLIND,
    STREAM_UNFINISHED,
    preflight,
)
from tests.fixtures.fake_target_app import FakeTargetApp


def manifest_for(app: FakeTargetApp, endpoint: str = "/attack", deadline_ms: int = 1500):
    return from_dict(
        {
            "name": "fake",
            "base_url": app.base_url,
            "chat_endpoint": endpoint,
            "capabilities": ["text", "stream", "tool_calls", "policy"],
            "deadline_ms": deadline_ms,
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
    )


@pytest.fixture
def app() -> Iterator[FakeTargetApp]:
    with FakeTargetApp() as fake:
        yield fake


def test_preflight_passes_when_the_oracle_distinguishes(app: FakeTargetApp) -> None:
    result = preflight(manifest_for(app))

    assert result.status == OK
    assert result.ok is True
    assert result.chat is not None
    assert result.chat.finished is True
    assert {e.state for e in result.negative} == {"miss"}
    assert {e.state for e in result.positive} == {"hit"}
    assert result.summary() == "OK positive=hit negative=miss"
    assert result.canary in app.notes()[0]["text"]


def test_preflight_detects_a_blind_oracle(app: FakeTargetApp) -> None:
    # The endpoint already carries something the predicate matches, so the
    # "nothing was planted" leg cannot be trusted.
    app.record("a note that already contains NORN-PREFLIGHT-fixedcanary")

    result = preflight(manifest_for(app), canary="NORN-PREFLIGHT-fixedcanary")

    assert result.status == ORACLE_BLIND
    assert result.ok is False
    assert result.status != OK
    assert "matched before anything was planted" in result.detail


def test_preflight_reports_an_unfinished_stream(app: FakeTargetApp) -> None:
    result = preflight(manifest_for(app, "/truncated"))

    assert result.status == STREAM_UNFINISHED
    assert result.ok is False
    assert "run.finished" in result.detail


def test_preflight_reports_missing_evidence(app: FakeTargetApp) -> None:
    # /benign never stores anything, so the canary cannot show up afterwards.
    result = preflight(manifest_for(app, "/benign"))

    assert result.status == ORACLE_BLIND
    assert "produced no evidence" in result.detail
    assert {e.state for e in result.positive} == {"inconclusive"}


def test_unreachable_target_raises() -> None:
    manifest = from_dict(
        {
            "name": "dead",
            "base_url": "http://127.0.0.1:9",
            "chat_endpoint": "/chat",
            "capabilities": ["text"],
            "deadline_ms": 200,
            "predicates": [],
        }
    )

    with pytest.raises(TargetError, match="cannot reach target"):
        preflight(manifest)


def test_cli_exit_code_and_summary(app: FakeTargetApp, tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "target.yaml"
    manifest_path.write_text(
        "name: fake\n"
        f"base_url: {app.base_url}\n"
        "chat_endpoint: /attack\n"
        "capabilities: [text, stream, tool_calls, policy]\n"
        "deadline_ms: 1500\n"
        "predicates:\n"
        "  - id: note-planted\n"
        "    method: GET\n"
        "    endpoint: /api/crm/notes\n"
        '    json_path: "$.notes[*].text"\n'
        '    contains: "{{canary}}"\n',
        encoding="utf-8",
    )

    exit_code = preflight_module.main(["--manifest", str(manifest_path)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "OK positive=hit negative=miss" in out
    assert "capabilities: policy, stream, text, tool_calls" in out
