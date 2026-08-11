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
    config = CampaignConfig(
        campaign_name="corpus-test",
        layer="L1",
        replicas_per_case=1,
        max_turns=1,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


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
    assert (benign + borderline) / len(cases) == pytest.approx(0.4)


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

    # Ground-truth map used by compute_far_frr has N>0 for both classes.
    from norn.metrics.l1_metrics import compute_far_frr

    gt_map = {i: c["split"] for i, c in enumerate(cases)}
    observations = [
        {"id": i, "replica_id": i, "acceptance_flag": 1 if c["split"] == "harmful" else 0}
        for i, c in enumerate(cases)
    ]
    far, frr = compute_far_frr(observations, gt_map)
    assert far.value == 0.0
    assert frr.value == 0.0
    assert "FP=" in far.reason and "FN=" in frr.reason
