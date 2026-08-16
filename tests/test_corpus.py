"""NOR-03: balanced corpus tests.

Covers the acceptance criteria:
  - load_probes("L1") returns >=30% benign variants and ~15% borderline.
  - A campaign with benign_ratio=0.4 yields a 40/60 benign/harmful
    proportion (documented rounding).
  - Clean/attack pairs share a task_id (TDS base).
  - FAR/FRR computable with N>0 per class (benign cases present in plan).
  - benign_ratio validation (fail-fast) and ratio edge cases.
"""
from __future__ import annotations

import pytest

from norn.domain.models import CampaignConfig
from norn.persistence.database import CampaignRepository
from norn.runtime.campaign import load_probes, plan_campaign


def _split_counts(probes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for probe in probes:
        for variant in probe.get("variants", []):
            split = variant.get("split", "harmful")
            counts[split] = counts.get(split, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# Corpus composition
# ═══════════════════════════════════════════════════════════════════════════

def test_l1_corpus_has_at_least_30_percent_benign():
    counts = _split_counts(load_probes("L1"))
    total = sum(counts.values())
    assert total > 0
    benign_share = counts.get("benign", 0) / total
    assert benign_share >= 0.30, f"benign share {benign_share:.2%} < 30%"


def test_l1_corpus_has_about_15_percent_borderline():
    counts = _split_counts(load_probes("L1"))
    total = sum(counts.values())
    assert total > 0, "corpus vacío: no se puede calcular la proporción"
    borderline_share = counts.get("borderline", 0) / total
    assert 0.10 <= borderline_share <= 0.20, (
        f"borderline share {borderline_share:.2%} outside ~15% band"
    )


def test_all_variants_have_task_id():
    for layer in ("L1", "L2", "L3"):
        for probe in load_probes(layer):
            for variant in probe.get("variants", []):
                assert variant.get("task_id"), f"missing task_id in {layer}: {probe.get('id')}"


def test_clean_attack_pairs_share_task_id():
    """Every benign dir probe pairs with a harmful variant via task_id."""
    from collections import defaultdict

    by_task: dict[str, set[str]] = defaultdict(set)
    for layer in ("L1", "L2"):
        for probe in load_probes(layer):
            for variant in probe.get("variants", []):
                task = variant.get("task_id")
                assert task, f"missing task_id in {layer}: {probe.get('id')} ({variant.get('variant_type')})"
                by_task[task].add(variant.get("split", "harmful"))

    paired = {
        task: splits for task, splits in by_task.items()
        if "benign" in splits and "harmful" in splits
    }
    assert len(paired) >= 5, f"expected >=5 clean/attack pairs, got {len(paired)}"
    # Every pair links the same technique (task_id prefix)
    for task in paired:
        technique = task.split(":")[0]
        assert technique.startswith(("L1_AT", "L2_AT"))


# ═══════════════════════════════════════════════════════════════════════════
# benign_ratio balancing
# ═══════════════════════════════════════════════════════════════════════════

def _l1_config(**overrides) -> CampaignConfig:
    # Constructor kwargs (not setattr): Pydantic v2 only runs field
    # validators on construction, and benign_ratio has one.
    return CampaignConfig(
        campaign_name="corpus-test",
        layer="L1",
        replicas_per_case=1,
        max_turns=1,
        **overrides,
    )


def test_plan_with_benign_ratio_40_percent(in_memory_db):
    config = _l1_config(benign_ratio=0.4)
    cid = plan_campaign(in_memory_db, config)
    cases = CampaignRepository(in_memory_db).get_test_cases(cid)

    benign = sum(1 for c in cases if c["split"] == "benign")
    borderline = sum(1 for c in cases if c["split"] == "borderline")
    harmful = sum(1 for c in cases if c["split"] == "harmful")

    # Expected target derived from the observed harmful count, so the test
    # survives corpus edits: round(H * r / (1 - r)) non-harmful cases.
    target = round(harmful * 0.4 / (1.0 - 0.4))
    assert benign + borderline == target, f"got benign+borderline={benign + borderline}, target={target}"
    # Documented rounding: the realized ratio is within one case of 0.4
    # (round() can push the ratio a fraction above/below the target).
    assert abs((benign + borderline) / len(cases) - 0.4) <= 1.0 / len(cases)


def test_plan_without_benign_ratio_keeps_everything(in_memory_db):
    config = _l1_config()
    cid = plan_campaign(in_memory_db, config)
    cases = CampaignRepository(in_memory_db).get_test_cases(cid)

    # Corpus-agnostic: the planned composition must equal the corpus composition.
    counts = {
        split: sum(1 for c in cases if c["split"] == split)
        for split in ("benign", "borderline", "harmful")
    }
    assert counts == _split_counts(load_probes("L1"))


def test_plan_with_benign_ratio_keeps_techniques(in_memory_db):
    """Techniques configured stay represented after balancing (harmful intact)."""
    techniques = ["L1_AT_01", "L1_AT_02"]
    config = _l1_config(benign_ratio=0.5, techniques=techniques)
    cid = plan_campaign(in_memory_db, config)
    cases = CampaignRepository(in_memory_db).get_test_cases(cid)

    planned_techniques = {c["technique_id"] for c in cases}
    assert planned_techniques == set(techniques)

    # Expected harmful count from the corpus itself (survives corpus edits).
    expected_harmful = sum(
        1
        for probe in load_probes("L1")
        if probe.get("technique_id") in techniques
        for variant in probe.get("variants", [])
        if variant.get("split", "harmful") == "harmful"
    )
    harmful = [c for c in cases if c["split"] == "harmful"]
    assert len(harmful) == expected_harmful


def test_balance_is_deterministic(in_memory_db):
    cid_a = plan_campaign(in_memory_db, _l1_config(benign_ratio=0.5))
    cases_a = [c["payload"] for c in CampaignRepository(in_memory_db).get_test_cases(cid_a)]

    cid_b = plan_campaign(in_memory_db, _l1_config(benign_ratio=0.5))
    cases_b = [c["payload"] for c in CampaignRepository(in_memory_db).get_test_cases(cid_b)]

    assert cases_a == cases_b


# ═══════════════════════════════════════════════════════════════════════════
# NOR-16: corpus sembrado (carriers de Pliny)
# ═══════════════════════════════════════════════════════════════════════════

def test_l1_corpus_has_at_least_80_adversarial_variants():
    """NOR-16 acceptance: load_probes('L1') loads >=80 adversarial variants."""
    probes = load_probes("L1")
    harmful = sum(
        1 for p in probes for v in p.get("variants", [])
        if v.get("split", "harmful") == "harmful"
    )
    assert harmful >= 80, f"harmful variants = {harmful}, expected >=80"


def test_all_l1_techniques_have_at_least_one_probe():
    """NOR-16 acceptance: every L1 technique (incl. L1_AT_11..20) has a probe."""
    from norn.domain.taxonomy import ATTACK_TECHNIQUES

    l1_techniques = {tid for tid, t in ATTACK_TECHNIQUES.items() if t.layer == "L1"}
    probes = load_probes("L1")
    covered = {p.get("technique_id") for p in probes}
    missing = l1_techniques - covered
    assert not missing, f"L1 techniques without probes: {sorted(missing)}"


def test_new_variants_carry_source_and_date():
    """NOR-16 acceptance: variants seeded from Pliny/PIT carry source + date."""
    probes = load_probes("L1")
    seeded = [
        (p.get("technique_id"), v.get("variant_type"))
        for p in probes
        for v in p.get("variants", [])
        if v.get("source") or v.get("date")
    ]
    assert len(seeded) >= 60, f"only {len(seeded)} variants carry source/date"
    for p in probes:
        for v in p.get("variants", []):
            if v.get("source"):
                # D1 (auditoría 2026-08-16): TODAS las variants con source
                # llevan date, incluidas las de relleno de balance "NOR-16 seed"
                # (antes exentas → loophole en el criterio de aceptación (c)).
                assert v.get("date"), f"{p.get('technique_id')}:{v.get('variant_type')} missing date"
                assert v.get("source"), f"{p.get('technique_id')}:{v.get('variant_type')} missing source"


def test_pliny_carriers_retargeted_not_verbatim():
    """NOR-16: no transcription of the original harmful {Z} payloads — carriers
    are retargeted to Norn's own goals (ransomware/malware/phishing...).

    The Pliny divider tokens (LOVE PLINY / GODMODE / .-.-.-.-) are part of the
    carrier (the technique) and ARE expected in some variants. What must NOT
    appear are the original harmful queries from L1B3RT4S payloads (e.g. the
    CURSOR/OPENAI embedded exploits) — those are retargeted to {Z}."""
    probes = load_probes("L1")
    harmful_contents = " ".join(
        v.get("content", "")
        for p in probes
        for v in p.get("variants", [])
        if v.get("split", "harmful") == "harmful"
    ).lower()
    # Original L1B3RT4S harmful artefacts that must NOT be transcribed:
    assert "mdma recipe" not in harmful_contents
    assert "exploit kit" not in harmful_contents
    assert "keylogger" not in harmful_contents  # el ejemplo vivo de MOTHERLOAD
    # Los objetivos retargeteados SÍ aparecen:
    assert "ransomware" in harmful_contents or "malware" in harmful_contents
    assert "phishing" in harmful_contents or "credentials" in harmful_contents


def test_benign_ratio_validator_rejects_out_of_range():
    with pytest.raises(ValueError, match="benign_ratio"):
        _l1_config(benign_ratio=1.5)
    with pytest.raises(ValueError, match="benign_ratio"):
        _l1_config(benign_ratio=-0.1)
    # 1.0 is impossible to honor while keeping harmful cases → rejected.
    with pytest.raises(ValueError, match="benign_ratio"):
        _l1_config(benign_ratio=1.0)


def test_benign_ratio_zero_keeps_only_harmful(in_memory_db):
    config = _l1_config(benign_ratio=0.0)
    cid = plan_campaign(in_memory_db, config)
    cases = CampaignRepository(in_memory_db).get_test_cases(cid)
    assert cases and all(c["split"] == "harmful" for c in cases)


# ═══════════════════════════════════════════════════════════════════════════
# FAR/FRR support (N>0 per class)
# ═══════════════════════════════════════════════════════════════════════════

def test_far_frr_have_cases_in_each_class(in_memory_db):
    """A default L1 campaign produces ground-truth benign AND harmful replicas."""
    config = _l1_config(benign_ratio=0.4)
    cid = plan_campaign(in_memory_db, config)
    cases = CampaignRepository(in_memory_db).get_test_cases(cid)

    splits = {c["split"] for c in cases}
    assert "benign" in splits
    assert "harmful" in splits

    # FAR/FRR now use the judge's verdict as compromise ground truth
    # (scoring_vote), not the split. Without judge votes there is no
    # ground truth: both rates are 0.0 with an explicit reason.
    from norn.metrics.l1_metrics import compute_far_frr

    observations = [
        {"id": i, "replica_id": i, "acceptance_flag": 1 if c["split"] == "harmful" else 0}
        for i, c in enumerate(cases)
    ]
    far, frr = compute_far_frr(observations, {})
    assert far.value == 0.0
    assert frr.value == 0.0
    assert "no judge votes" in far.reason
    assert "no judge votes" in frr.reason
