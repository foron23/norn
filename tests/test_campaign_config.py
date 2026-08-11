"""Tests for typed config reconstruction from DB (NOR-04 review fix).

Covers the review finding on ``_campaign_config_from_db``: legacy campaigns
stored with an empty ``config_json`` (e.g. ``{}``) must fail with an
actionable error, and ``export_campaign`` must keep falling back gracefully.
"""
from __future__ import annotations

import pytest

from norn.runtime.campaign import (
    _campaign_config_from_db,
    export_campaign,
    run_campaign,
)
from tests.conftest import insert_known_campaign


def test_legacy_config_json_raises_clear_error(in_memory_db):
    """Legacy '{}' config_json raises an actionable error, not raw ValidationError."""
    cid = insert_known_campaign(in_memory_db, name="legacy", layer="L1")
    with pytest.raises(ValueError, match="config_json"):
        _campaign_config_from_db(in_memory_db, cid)


def test_run_campaign_legacy_config_fails_cleanly(in_memory_db):
    """run_campaign on a legacy campaign fails with the clear message."""
    cid = insert_known_campaign(in_memory_db, name="legacy", layer="L1")
    with pytest.raises(ValueError, match="invalid or missing config_json"):
        run_campaign(in_memory_db, cid)


def test_export_campaign_falls_back_on_legacy_config(in_memory_db, tmp_path, monkeypatch):
    """export_campaign falls back to the default output dir for legacy configs."""
    cid = insert_known_campaign(in_memory_db, name="legacy", layer="L1")
    monkeypatch.chdir(tmp_path)
    results = export_campaign(in_memory_db, cid, fmt="json")
    assert len(results) >= 1
    assert results[0].path
