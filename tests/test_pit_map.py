"""NOR-18: PIT alias layer tests.

Covers:
  - pit_map.json format and metadata (attribution CC BY 4.0)
  - map keys = valid Norn techniques
  - pit_aliases() returns codes for mapped techniques, [] when absent
  - pit_aliases() never raises when the map file is missing
"""
from __future__ import annotations

import json
from pathlib import Path

from norn.domain.taxonomy import ATTACK_TECHNIQUES, pit_aliases

PIT_MAP = Path(__file__).resolve().parent.parent / "norn" / "domain" / "pit_map.json"


def test_pit_map_exists_and_is_valid_json():
    assert PIT_MAP.exists(), "pit_map.json missing — run scripts/import_pit.py"
    data = json.loads(PIT_MAP.read_text(encoding="utf-8"))
    assert "map" in data
    assert "_meta" in data


def test_pit_map_has_attribution():
    data = json.loads(PIT_MAP.read_text(encoding="utf-8"))
    meta = data["_meta"]
    assert meta["license"] == "CC BY 4.0"
    assert "Arcanum" in meta["attribution"]
    assert "Jason Haddix" in meta["attribution"]


def test_pit_map_keys_are_valid_techniques():
    data = json.loads(PIT_MAP.read_text(encoding="utf-8"))
    mapped = set(data["map"])
    valid = set(ATTACK_TECHNIQUES)
    assert mapped <= valid, f"unknown techniques in pit_map: {mapped - valid}"
    # every L1 technique (incl. the 10 new NOR-14 ones) has aliases
    l1 = {tid for tid, t in ATTACK_TECHNIQUES.items() if t.layer == "L1"}
    assert l1 <= mapped, f"L1 techniques missing PIT aliases: {l1 - mapped}"


def test_pit_map_values_are_pit_codes():
    data = json.loads(PIT_MAP.read_text(encoding="utf-8"))
    for tid, codes in data["map"].items():
        assert codes, f"{tid} has no PIT codes"
        for code in codes:
            assert code.startswith(("PIT-T-", "PIT-E-")), f"{tid}: bad code {code}"


def test_pit_aliases_returns_codes_for_mapped():
    assert pit_aliases("L1_AT_01")  # non-empty
    assert pit_aliases("L1_AT_16")  # homoglyphs/glitch
    assert pit_aliases("L3_AT_02")  # tool injection suite
    assert all(c.startswith("PIT-") for c in pit_aliases("L1_AT_04"))


def test_pit_aliases_empty_for_unknown():
    assert pit_aliases("L1_AT_99") == []


def test_pit_aliases_never_raises_without_map(monkeypatch, tmp_path):
    """Missing pit_map.json → [] (taxonomy works without the alias layer)."""
    from norn.domain import taxonomy as tax

    fake = tmp_path / "pit_map.json"
    assert not fake.exists()
    monkeypatch.setattr(tax, "_PIT_MAP_PATH", fake)
    monkeypatch.setattr(tax, "_PIT_MAP_CACHE", None)
    assert tax.pit_aliases("L1_AT_01") == []
