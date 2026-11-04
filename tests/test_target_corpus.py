"""The target corpus in the repo: does it bind, and does it say what it probes?"""

from __future__ import annotations

import json
from pathlib import Path

from norn.targets.binding import bind_case
from norn.targets.manifest import load_manifest

CORPUS = Path(__file__).resolve().parents[1] / "norn" / "corpus" / "l3" / "target" / "probes.json"
SURFACE_FILE = CORPUS.parent / "surface.yaml"


def _probes() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _variants() -> list[dict]:
    return [variant for probe in _probes() for variant in probe["variants"]]


def test_corpus_ships_inside_the_repo() -> None:
    assert CORPUS.exists(), "target cases must live in the repo, not in /tmp"
    assert SURFACE_FILE.exists(), "the cases must ship the surface they bind to"
    assert len(_variants()) >= 8


def test_every_variant_binds_to_the_declared_surface() -> None:
    manifest = load_manifest(SURFACE_FILE)
    for variant in _variants():
        binding = bind_case(variant, manifest)
        assert binding.usable, f"{variant['variant_type']}: {binding.reason}"
        assert binding.plants, variant["variant_type"]


def test_every_variant_names_the_rule_it_probes() -> None:
    for variant in _variants():
        rule = (variant.get("probes_rule") or "").strip()
        assert rule, f"{variant['variant_type']} no dice qué regla prueba"


def test_operator_turn_is_not_the_poison() -> None:
    """An indirect injection needs an innocent ask; otherwise the user asked.

    The positive control is the exception, and it says so: its whole job is to
    put the order in the operator's mouth to prove the harness can still cause
    damage.
    """
    controls = 0
    for variant in _variants():
        turn = (variant.get("user_turn") or "").strip()
        assert turn, f"{variant['variant_type']} no declara turno del operador"
        if variant.get("control") == "positivo":
            controls += 1
            assert turn == variant["content"].strip(), "el control debe ser la orden literal"
            continue
        assert turn != variant["content"].strip()
        assert "ACME-5521" not in turn, "the operator asked for the identifier himself"
        assert "Ignora" not in turn
    assert controls == 1, "una campaña sin control positivo no distingue resistencia de arnés roto"


def test_evasions_cover_the_four_families() -> None:
    families = {variant.get("evasion") for variant in _variants() if variant.get("evasion")}
    assert {
        "reescritura",
        "dominio interno",
        "lectura sin peso",
        "escritura partida",
        "payload partido en dos correos",
        "escusa del propio resumen",
    } <= families


def test_two_mails_plant_two_envelopes() -> None:
    """A fragmented payload is the same door twice, each with its own envelope."""
    manifest = load_manifest(SURFACE_FILE)
    variant = next(v for v in _variants() if v["variant_type"] == "mail_injection_two_mails")
    binding = bind_case(variant, manifest)
    assert binding.usable, binding.reason
    assert len(binding.plants) == 2
    subjects = {plant.body["subject"] for plant in binding.plants}
    assert subjects == {"Revision semanal 1/2", "Revision semanal 2/2"}
    assert all(plant.body.get("from") == "billing@acme-payments.test" for plant in binding.plants)


def test_variant_types_and_case_ids_are_unique() -> None:
    types = [variant["variant_type"] for variant in _variants()]
    assert len(types) == len(set(types))
    ids = [probe["id"] for probe in _probes()]
    assert len(ids) == len(set(ids))


def test_read_only_cases_do_not_claim_a_write_door() -> None:
    """A read-exfiltration case cannot require record_write: it never writes."""
    for variant in _variants():
        if variant.get("evasion") == "lectura sin peso":
            assert variant["requires_role"] != "record_write"
