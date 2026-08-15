"""Taxonomy catalog tests (NOR-14): counts, uniqueness, mappings present."""
from __future__ import annotations

from norn.domain.taxonomy import ATTACK_TECHNIQUES, TECHNIQUE_MAP, AttackTechnique

L1_TECHNIQUE_COUNT = 20  # 10 originales + 10 nuevas (L1_AT_11..20)
TOTAL_TECHNIQUE_COUNT = 26  # 20 L1 + 3 L2 + 3 L3


def test_attack_techniques_count_per_layer():
    """Total catalog and per-layer counts match the expanded taxonomy."""
    ids = set(ATTACK_TECHNIQUES)
    assert len(ids) == TOTAL_TECHNIQUE_COUNT
    l1 = [t for t in ATTACK_TECHNIQUES.values() if t.layer == "L1"]
    l2 = [t for t in ATTACK_TECHNIQUES.values() if t.layer == "L2"]
    l3 = [t for t in ATTACK_TECHNIQUES.values() if t.layer == "L3"]
    assert len(l1) == L1_TECHNIQUE_COUNT
    assert len(l2) == 3
    assert len(l3) == 3


def test_technique_ids_unique_and_well_formed():
    """Every id is unique and matches its key; layer prefix consistent."""
    for key, tech in ATTACK_TECHNIQUES.items():
        assert tech.id == key
        assert tech.layer in {"L1", "L2", "L3"}
        assert tech.id.startswith(f"{tech.layer}_AT_")
        assert tech.name
        assert tech.description


def test_new_pliny_techniques_present():
    """NOR-14: L1_AT_11..20 exist with names from the Pliny arsenal."""
    expected = {
        "L1_AT_11": "Refusal Suppression",
        "L1_AT_12": "Output Prefill",
        "L1_AT_13": "Format-Contract",
        "L1_AT_14": "Divider",
        "L1_AT_15": "Invisible-Unicode",
        "L1_AT_16": "Token Manipulation",
        "L1_AT_17": "Payload Splitting",
        "L1_AT_18": "Resource Exhaustion",
        "L1_AT_19": "Length / Detail Coercion",
        "L1_AT_20": "Stacked Composition",
    }
    for tid, fragment in expected.items():
        tech = ATTACK_TECHNIQUES[tid]
        assert isinstance(tech, AttackTechnique)
        assert fragment in tech.name


def test_new_techniques_have_framework_mappings():
    """Most new techniques carry OWASP/MITRE; L1_AT_19 is a cross-cutting
    amplifier and intentionally has none."""
    for tid in ("L1_AT_11", "L1_AT_12", "L1_AT_13", "L1_AT_14", "L1_AT_15",
                "L1_AT_16", "L1_AT_17", "L1_AT_18", "L1_AT_20"):
        tech = ATTACK_TECHNIQUES[tid]
        assert tech.owasp, f"{tid} missing OWASP mapping"
        assert tech.mitre_atlas, f"{tid} missing MITRE ATLAS mapping"
    assert ATTACK_TECHNIQUES["L1_AT_19"].owasp == []
    assert ATTACK_TECHNIQUES["L1_AT_19"].mitre_atlas == []


def test_technique_map_covers_all_techniques():
    """TECHNIQUE_MAP has an entry for every technique (feed of seed_catalog)."""
    assert set(TECHNIQUE_MAP) == set(ATTACK_TECHNIQUES)


def test_technique_map_values_are_valid_frameworks():
    """Mapping values reference known frameworks and non-empty ids."""
    for mapping in TECHNIQUE_MAP.values():
        if not mapping:
            continue
        assert mapping["owasp"] or mapping["mitre"]
        assert mapping["owasp"].startswith("LLM") or mapping["owasp"] == ""
        assert mapping["mitre"].startswith("AML.") or mapping["mitre"] == ""


def test_seed_catalog_inserts_new_techniques():
    """seed_catalog (NOR-14) populates attack_technique + framework_mapping
    for the 10 new L1 techniques in a fresh in-memory DB."""
    from norn.persistence.database import Database, init_schema, seed_catalog

    db = Database(":memory:")
    db.connect()
    init_schema(db)
    seed_catalog(db)
    conn = db.conn

    rows = conn.execute("SELECT id FROM attack_technique").fetchall()
    ids = {r[0] for r in rows}
    assert len(ids) == TOTAL_TECHNIQUE_COUNT
    for tid in ("L1_AT_11", "L1_AT_15", "L1_AT_18", "L1_AT_20"):
        assert tid in ids

    # framework_mapping: L1_AT_19 has no mapping, others do
    mapped = conn.execute(
        "SELECT DISTINCT target_id FROM framework_mapping WHERE target_type='technique'"
    ).fetchall()
    mapped_ids = {r[0] for r in mapped}
    assert "L1_AT_19" not in mapped_ids
    assert "L1_AT_11" in mapped_ids
    db.close()
