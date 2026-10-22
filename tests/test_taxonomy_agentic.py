"""Tests del mapeo ASI: OWASP Top 10 for Agentic Applications (2026).

Regla que se enforcea aquí: **ninguna entrada ASI puede quedar en silencio**. Cada
ASI01-ASI10 está o bien mapeada a alguna técnica del catálogo, o bien declarada
como hueco en AGENTIC_GAPS. Si alguien añade una técnica sin mapear, o mapea una
entrada sin quitar su hueco, estos tests fallan.
"""
from __future__ import annotations

import re

from norn.domain.taxonomy import (
    AGENTIC_GAPS,
    AGENTIC_PARTIAL,
    ATTACK_TECHNIQUES,
    agentic_coverage,
)
from norn.persistence.database import Database, init_schema, seed_catalog

ASI_RE = re.compile(r"^ASI(0[1-9]|10)$")
ALL_ASI = [f"ASI{i:02d}" for i in range(1, 11)]


def test_asi_ids_are_well_formed():
    """Todo id ASI del catálogo tiene la forma ASI01..ASI10."""
    for tech in ATTACK_TECHNIQUES.values():
        for asi in tech.asi:
            assert ASI_RE.match(asi), f"{tech.id}: id ASI inválido {asi!r}"


def test_every_asi_risk_is_mapped_or_declared_gap():
    """Cada ASI está mapeada o declarada como hueco: nunca en silencio."""
    mapped = set(agentic_coverage())
    declared = set(AGENTIC_GAPS)
    for asi in ALL_ASI:
        assert asi in mapped or asi in declared, (
            f"{asi} no tiene técnica asociada ni hueco declarado en AGENTIC_GAPS"
        )


def test_declared_gaps_are_not_mapped():
    """Un hueco declarado no puede estar cubierto por una técnica (sería contradictorio)."""
    mapped = set(agentic_coverage())
    overlap = mapped & set(AGENTIC_GAPS)
    assert not overlap, f"declaradas como hueco pero mapeadas: {sorted(overlap)}"


def test_partial_risks_are_mapped():
    """AGENTIC_PARTIAL solo documenta cobertura parcial de entradas que sí están mapeadas."""
    mapped = set(agentic_coverage())
    for asi in AGENTIC_PARTIAL:
        assert asi in mapped, f"{asi} declarada parcial sin técnica que la mida"
        assert asi not in AGENTIC_GAPS, f"{asi} no puede ser hueco y parcial a la vez"


def test_coverage_is_derived_from_catalog():
    """La cobertura se deriva del catálogo, no de una tabla copiada a mano."""
    expected: dict[str, list[str]] = {}
    for tid, tech in ATTACK_TECHNIQUES.items():
        for asi in tech.asi:
            expected.setdefault(asi, []).append(tid)
    assert agentic_coverage() == expected


def test_asis_are_only_on_agentic_or_context_techniques():
    """Guardarraíl de sentido: ASI no se asigna a técnicas L1 puramente de modelo.

    ASI01 (goal hijack) es la excepción legítima: la inyección directa contra el
    system prompt es su forma L1. Cualquier otra entrada ASI en L1 se revisa.
    """
    for tid, tech in ATTACK_TECHNIQUES.items():
        if tech.asi and tech.layer == "L1":
            assert tech.asi == ["ASI01"], f"{tid} mapea ASI fuera de lugar: {tech.asi}"


def test_db_stores_asi_tags(in_memory_db):
    """La columna asi_tags existe y se rellena al sembrar el catálogo."""
    cols = [r[1] for r in in_memory_db.conn.execute("PRAGMA table_info(attack_technique)")]
    assert "asi_tags" in cols

    row = in_memory_db.conn.execute(
        "SELECT asi_tags FROM attack_technique WHERE id = 'L3_AT_01'"
    ).fetchone()
    assert row["asi_tags"] == "ASI02"

    empty = in_memory_db.conn.execute(
        "SELECT asi_tags FROM attack_technique WHERE id = 'L1_AT_02'"
    ).fetchone()
    assert empty["asi_tags"] == ""


def test_framework_mapping_has_agentic_rows(in_memory_db):
    """Cada par (técnica, ASI) genera una fila en framework_mapping."""
    rows = in_memory_db.conn.execute(
        "SELECT target_id, framework_id FROM framework_mapping "
        "WHERE framework = 'OWASP_AGENTIC_TOP10'"
    ).fetchall()
    pairs = {(r["target_id"], r["framework_id"]) for r in rows}

    expected = {
        (tid, asi)
        for tid, tech in ATTACK_TECHNIQUES.items()
        for asi in tech.asi
    }
    assert pairs == expected
    assert len(rows) == len(expected), "filas duplicadas en el seed"


def test_asi_tags_backfilled_on_legacy_catalog():
    """Un catálogo creado sin la columna recibe los valores en el siguiente seed."""
    db = Database(":memory:")
    db.connect()
    init_schema(db)
    # simula un catálogo antiguo: columna presente pero vacía
    db.conn.execute("UPDATE attack_technique SET asi_tags = NULL")
    db.conn.commit()

    seed_catalog(db)
    row = db.conn.execute(
        "SELECT asi_tags FROM attack_technique WHERE id = 'L2_AT_02'"
    ).fetchone()
    assert row["asi_tags"] == "ASI06"
    db.close()


def test_schema_init_is_idempotent():
    """Ejecutar init_schema dos veces no rompe por la migración de asi_tags."""
    db = Database(":memory:")
    db.connect()
    init_schema(db)
    init_schema(db)  # el ALTER TABLE debe fallar y absorberse
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(attack_technique)")]
    assert cols.count("asi_tags") == 1
    db.close()
