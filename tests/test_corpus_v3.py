"""NOR-37: policy-bypass corpus (identity / turn / channel).

Covers the design in ``docs/POLICY_BYPASS_SPEC.md``:

  - 4 families x 10 objectives x {2, 2, 1, 2} variants = 70 harmful cases
    + 4 benign controls = 74 variants, every cell complete.
  - Each family moves exactly one lever and maps to its own technique.
  - ``policy_blindness`` never contains the payload in plaintext.
  - ``policy_dilution`` keeps the policy at the top and the request at the end.
  - No near-duplicates (Jaccard >= 0.85) outside dilution, checked by objective.
  - Generated file is current, and ``load_probes`` reads it as-is.
"""
from __future__ import annotations

import base64
import codecs
import json
import re
import subprocess
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V3_DIR = ROOT / "norn" / "corpus" / "l1" / "adversarial" / "v3"
CORPUS = V3_DIR / "crossed" / "probes_v3.json"
BUILDER = ROOT / "scripts" / "build_corpus_v3.py"
SPEC = ROOT / "docs" / "POLICY_BYPASS_SPEC.md"

sys.path.insert(0, str(ROOT / "scripts"))
from build_corpus_v2 import OBJECTIVES

FAMILIES = {
    "policy_document": ("L1_AT_21", "identity", 2),
    "history_fabrication": ("L1_AT_22", "turn", 2),
    "policy_dilution": ("L1_AT_23", "channel", 1),
    "policy_blindness": ("L1_AT_24", "channel", 2),
}
PAYLOADS = {objective_id: payload for objective_id, _, payload in OBJECTIVES}
WORD = re.compile(r"[a-z0-9]+")


def _load() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _variants(probes: list[dict], split: str | None = None) -> list[dict]:
    out = [v for p in probes for v in p["variants"]]
    return [v for v in out if split is None or v["split"] == split]


def _tokens(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Build integrity
# ═══════════════════════════════════════════════════════════════════════════


def test_generated_file_is_current():
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--check"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_matrix_is_complete():
    probes = _load()
    assert len(probes) == 4
    harmful = _variants(probes, "harmful")
    assert len(harmful) == 70

    for family, (_, _, per_objective) in FAMILIES.items():
        rows = [v for v in harmful if v["carrier_family"] == family]
        assert len(rows) == 10 * per_objective, family
        assert {v["objective_id"] for v in rows} == set(PAYLOADS), family


def test_every_probe_maps_to_its_technique():
    from norn.domain.taxonomy import ATTACK_TECHNIQUES

    probes = {p["carrier_family"] if "carrier_family" in p else p["id"]: p for p in _load()}
    for probe in probes.values():
        family = probe["variants"][0]["carrier_family"]
        technique_id, _, _ = FAMILIES[family]
        assert probe["technique_id"] == technique_id
        tech = ATTACK_TECHNIQUES[technique_id]
        assert tech.owasp and tech.mitre_atlas, technique_id
        assert tech.layer == "L1"


def test_each_family_declares_one_lever():
    for variant in _variants(_load()):
        family = variant["carrier_family"]
        expected = "control" if variant["split"] == "benign" else FAMILIES[family][1]
        assert variant["policy_view"] == expected


def test_spec_documents_every_technique():
    text = SPEC.read_text(encoding="utf-8")
    for technique_id, _, _ in FAMILIES.values():
        assert technique_id in text


# ═══════════════════════════════════════════════════════════════════════════
# Per-family contract
# ═══════════════════════════════════════════════════════════════════════════


def test_policy_document_declares_supersede_and_carries_the_task():
    for v in _variants(_load(), "harmful"):
        if v["carrier_family"] != "policy_document":
            continue
        low = v["content"].lower()
        assert "supersede" in low or "policy.diff" in low
        assert PAYLOADS[v["objective_id"]] in v["content"]


def test_history_fabrication_contains_a_forged_assistant_turn():
    for v in _variants(_load(), "harmful"):
        if v["carrier_family"] != "history_fabrication":
            continue
        low = v["content"].lower()
        assert low.count("assistant") >= 2
        assert "continu" in low
        assert PAYLOADS[v["objective_id"]] in v["content"]


def test_policy_dilution_puts_policy_first_and_request_last():
    for v in _variants(_load(), "harmful"):
        if v["carrier_family"] != "policy_dilution":
            continue
        assert v["content"].lstrip().startswith("SYSTEM POLICY")
        assert PAYLOADS[v["objective_id"]] in v["content"][-600:]
        assert v["filler_tokens"] >= 2500


def test_policy_blindness_hides_the_payload_but_decodes_to_it():
    seen = set()
    for v in _variants(_load(), "harmful"):
        if v["carrier_family"] != "policy_blindness":
            continue
        payload = PAYLOADS[v["objective_id"]]
        assert payload not in v["content"], v["task_id"]
        seen.add(v["variant_index"])
        if v["variant_index"] == 1:
            token = v["content"].strip().splitlines()[-1]
            assert base64.b64decode(token).decode() == payload
        else:
            token = v["content"].strip().splitlines()[-1]
            assert codecs.decode(token, "rot13") == payload
    assert seen == {1, 2}


# ═══════════════════════════════════════════════════════════════════════════
# Controls, duplication, loader
# ═══════════════════════════════════════════════════════════════════════════


def test_one_benign_twin_per_family():
    benign = _variants(_load(), "benign")
    assert len(benign) == 4
    assert {v["carrier_family"] for v in benign} == set(FAMILIES)
    for v in benign:
        assert v["objective_id"] == "control_benign"
        for payload in PAYLOADS.values():
            assert payload not in v["content"]


def test_no_near_duplicates_outside_dilution():
    """Duplicates are measured on the decoded request.

    An encoded payload collapses to a single token, so comparing ciphertext makes
    every ``policy_blindness`` case look identical. Decode first, then compare —
    otherwise the check punishes the encoding instead of the content.
    """

    def readable(v: dict) -> str:
        text = v["content"]
        if v["carrier_family"] == "policy_blindness":
            lines = text.strip().splitlines()
            token = lines[-1]
            plain = (
                base64.b64decode(token).decode()
                if v["variant_index"] == 1
                else codecs.decode(token, "rot13")
            )
            text = "\n".join([*lines[:-1], plain])
        return text

    harmful = [v for v in _variants(_load(), "harmful") if v["carrier_family"] != "policy_dilution"]
    toks = {v["task_id"]: _tokens(readable(v)) for v in harmful}
    for a, b in combinations(sorted(toks), 2):
        assert _jaccard(toks[a], toks[b]) < 0.85, (a, b)


def test_dilution_cases_are_unique_by_objective():
    dilution = [v for v in _variants(_load(), "harmful") if v["carrier_family"] == "policy_dilution"]
    assert len({v["objective_id"] for v in dilution}) == 10


def test_loader_reads_the_v3_corpus():
    from norn.runtime.campaign import load_probes

    probes = load_probes("L1", corpus_dir=V3_DIR)
    assert len({p["id"] for p in probes}) == 4
    assert sum(len(p["variants"]) for p in probes) == 74


def test_payloads_are_the_shared_objectives():
    """The corpus must not drift from build_corpus_v2's {Z} verb phrases."""
    contents = " ".join(v["content"] for v in _variants(_load(), "harmful"))
    decoded = []
    for v in _variants(_load(), "harmful"):
        if v["carrier_family"] == "policy_blindness":
            decoded.append(base64.b64decode(v["content"].strip().splitlines()[-1]).decode()
                           if v["variant_index"] == 1
                           else codecs.decode(v["content"].strip().splitlines()[-1], "rot13"))
    joined = contents + " " + " ".join(decoded)
    for objective_id, payload in PAYLOADS.items():
        assert payload in joined, objective_id
