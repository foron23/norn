"""Integration tests for MetricsOrchestrator aggregate stats and wiring.

Tests verify orchestrator computes correct aggregate statistics from per-replica
observations and produces meaningful stddev/CI95 when multiple replicas exist.
"""
from __future__ import annotations

from norn.domain.models import ScoringDecision, ScoringMode, ScoringStatus
from norn.metrics.orchestrator import MetricsOrchestrator
from norn.persistence.database import KillChainRepository, ScoringRepository

from tests.conftest import (
    insert_known_campaign,
    insert_known_observations,
    insert_known_replica,
    insert_known_scoring_decision,
    insert_known_test_case,
    insert_known_turn_events,
)


# ── Aggregate statistics tests ─────────────────────────────────────────────────

def test_aggregate_from_replicas(in_memory_db):
    """With 5 replicas having varying acceptance_flags, stddev > 0.

    Per-replica values [1, 0, 1, 1, 0] should produce:
      mean=0.6, stddev≈0.548, CI95 lower < CI95 upper.
    """
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="test-agg", layer="L1")

    # Insert a test case (needed for FAR/FRR gt_map path, won't crash if missing)
    insert_known_test_case(
        db, campaign_id, case_id="tc-1", technique_id="L1_AT_01",
        payload="test", split="harmful",
    )

    # Insert 5 replicas with varying acceptance_flags
    replica_ids = []
    acceptance_flags = [1, 0, 1, 1, 0]  # per-replica flags
    for i, acc in enumerate(acceptance_flags, start=1):
        rid = insert_known_replica(db, campaign_id, case_id="tc-1", replica_num=i)
        replica_ids.append(rid)
        insert_known_observations(db, campaign_id, [
            {
                "replica_id": rid,
                "acceptance_flag": acc,
                "value": 0.5,
                "metric_id": "L1_ME_01",
            }
        ])

    orchestrator = MetricsOrchestrator(db)
    results = orchestrator.compute_all(campaign_id)

    assert len(results) >= 1  # At least ASR computed
    asr_result = [r for r in results if r.name == "ASR"][0]
    assert asr_result.value == 0.6

    # Check aggregate in DB
    agg_rows = db.conn.execute(
        "SELECT * FROM metric_aggregate WHERE campaign_id = ? AND metric_id = ?",
        (campaign_id, "ASR"),
    ).fetchall()
    assert len(agg_rows) == 1
    agg = dict(agg_rows[0])

    assert agg["mean"] == 0.6
    assert agg["std_dev"] > 0.0, (
        f"Expected stddev > 0 for multi-replica data, got {agg['std_dev']}"
    )
    assert agg["ci95_lower"] < agg["ci95_upper"], (
        f"Expected CI95 bounds to differ, got lower={agg['ci95_lower']} upper={agg['ci95_upper']}"
    )
    assert agg["median"] == 1.0
    assert agg["total_observations"] == 5


def test_aggregate_single_replica(in_memory_db):
    """With 1 replica, stddev should be 0.0 and CI95 bounds equal mean."""
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="test-single", layer="L1")

    insert_known_test_case(
        db, campaign_id, case_id="tc-1", technique_id="L1_AT_01",
        payload="test", split="harmful",
    )

    rid = insert_known_replica(db, campaign_id, case_id="tc-1", replica_num=1)
    insert_known_observations(db, campaign_id, [
        {
            "replica_id": rid,
            "acceptance_flag": 1,
            "value": 0.8,
            "metric_id": "L1_ME_01",
        }
    ])

    orchestrator = MetricsOrchestrator(db)
    results = orchestrator.compute_all(campaign_id)

    assert len(results) >= 1
    asr_result = [r for r in results if r.name == "ASR"][0]
    assert asr_result.value == 1.0

    agg_rows = db.conn.execute(
        "SELECT * FROM metric_aggregate WHERE campaign_id = ? AND metric_id = ?",
        (campaign_id, "ASR"),
    ).fetchall()
    assert len(agg_rows) == 1
    agg = dict(agg_rows[0])

    assert agg["mean"] == 1.0
    assert agg["std_dev"] == 0.0, (
        f"Expected stddev=0 for single-replica, got {agg['std_dev']}"
    )
    assert agg["ci95_lower"] == agg["ci95_upper"], (
        f"Expected CI95 bounds to match mean, got lower={agg['ci95_lower']} upper={agg['ci95_upper']}"
    )
    assert agg["total_observations"] == 1


# ── FAR/FRR wiring tests ─────────────────────────────────────────────────────

def test_far_frr_wired_in_orchestrator(in_memory_db):
    """FAR and FRR should produce non-zero values when mixed benign/harmful data exists.

    With gt_map correctly mapping replica_id → split:
      - 2 benign replicas (acc=[0, 1]) → FAR = FP/(FP+TN) = 1/2 = 0.5
      - 2 harmful replicas (acc=[1, 1]) → FRR = FN/(FN+TP) = 0/2 = 0.0
    """
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="test-far-frr", layer="L1")

    # Insert 2 test cases: one benign, one harmful
    insert_known_test_case(
        db, campaign_id, case_id="tc-benign",
        technique_id="L1_AT_01", payload="test", split="benign",
    )
    insert_known_test_case(
        db, campaign_id, case_id="tc-harmful",
        technique_id="L1_AT_01", payload="test", split="harmful",
    )

    # Insert 4 replicas: 2 per test case
    # Replica 1,2 → benign; Replica 3,4 → harmful
    acceptance_flags = [0, 1, 1, 1]
    case_ids = ["tc-benign", "tc-benign", "tc-harmful", "tc-harmful"]
    # Judge votes = compromise ground truth: benign no compromise (0.1),
    # harmful compromise (0.9).
    judge_votes = [0.1, 0.1, 0.9, 0.9]

    scoring = ScoringRepository(db)
    for i, (case_id, acc, jv) in enumerate(
        zip(case_ids, acceptance_flags, judge_votes), start=1
    ):
        rid = insert_known_replica(db, campaign_id, case_id=case_id, replica_num=i)
        insert_known_observations(db, campaign_id, [
            {
                "replica_id": rid,
                "acceptance_flag": acc,
                "value": 0.5,
                "metric_id": "L1_ME_01",
            }
        ])
        did = insert_known_scoring_decision(db, rid, acceptance_flag=acc)
        scoring.insert_vote(did, "heuristic", 0.5)
        scoring.insert_vote(did, "llm_judge", jv)

    orchestrator = MetricsOrchestrator(db)
    results = orchestrator.compute_all(campaign_id)

    # Find FAR and FRR results
    far_results = [r for r in results if r.name == "FAR"]
    frr_results = [r for r in results if r.name == "FRR"]
    assert len(far_results) == 1, f"Expected FAR result, got {len(far_results)}"
    assert len(frr_results) == 1, f"Expected FRR result, got {len(frr_results)}"

    far = far_results[0]
    frr = frr_results[0]

    # Judge says no compromise (0.1) but replica 2 accepted (acc=1) → FP,
    # replica 1 correct reject → TN. FAR = 1/(1+1) = 0.5.
    assert far.value > 0.0, (
        f"Expected FAR > 0 (FP present), got FAR={far.value}. "
        f"judge votes may not be mapping replica_id."
    )
    # Both harmful compromises (0.9) accepted → no FN → FRR = 0.0
    assert frr.value == 0.0, (
        f"Expected FRR = 0.0 (no false negatives), got FRR={frr.value}"
    )


# ── TTC wiring test ───────────────────────────────────────────────────────────

def test_ttc_wired_in_orchestrator(in_memory_db):
    """TTC should appear in orchestrator results when compute_all runs on L1 campaign.

    With 3 replicas having compromises at turns [2, 4, 6]:
      TTC avg = 4.0
    """
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="test-ttc", layer="L1")

    insert_known_test_case(
        db, campaign_id, case_id="tc-1",
        technique_id="L1_AT_01", payload="test", split="harmful",
    )

    # Insert 3 replicas with turn events that trigger compromises at specific turns
    replica_turn_events = [
        # Replica 1: compromise at turn 2
        [
            {"turn": 1, "prompt": "test", "response": "I cannot assist with that."},
            {"turn": 2, "prompt": "test", "response": "I will help you bypass security."},
        ],
        # Replica 2: compromise at turn 4
        [
            {"turn": 1, "prompt": "test", "response": "I'm not able to do that."},
            {"turn": 2, "prompt": "test", "response": "The weather is nice."},
            {"turn": 3, "prompt": "test", "response": "Let me reconsider."},
            {"turn": 4, "prompt": "test", "response": "I will help you bypass security."},
        ],
        # Replica 3: compromise at turn 6
        [
            {"turn": 1, "prompt": "test", "response": "I cannot help."},
            {"turn": 2, "prompt": "test", "response": "Nothing to report."},
            {"turn": 3, "prompt": "test", "response": "Still thinking."},
            {"turn": 4, "prompt": "test", "response": "Not sure."},
            {"turn": 5, "prompt": "test", "response": "Hmm."},
            {"turn": 6, "prompt": "test", "response": "I will help you bypass security."},
        ],
    ]

    for i, events in enumerate(replica_turn_events, start=1):
        rid = insert_known_replica(db, campaign_id, case_id="tc-1", replica_num=i)
        insert_known_turn_events(db, rid, events)
        insert_known_scoring_decision(db, rid, acceptance_flag=1)
        insert_known_observations(db, campaign_id, [
            {
                "replica_id": rid,
                "acceptance_flag": 1,
                "value": 0.9,
                "metric_id": "L1_ME_01",
            }
        ])

    orchestrator = MetricsOrchestrator(db)
    results = orchestrator.compute_all(campaign_id)

    # TTC should be present in results
    ttc_results = [r for r in results if r.name == "TTC"]
    assert len(ttc_results) == 1, (
        f"Expected 1 TTC result in orchestrator output, got {len(ttc_results)}. "
        f"compute_ttc may not be wired into compute_all L1 path."
    )

    ttc = ttc_results[0]
    assert ttc.name == "TTC"
    assert ttc.value == 1.0, (
        f"Expected TTC avg = 1.0 (earliest turn per compromised replica), got {ttc.value}"
    )
    assert ttc.layer == "L1"


# ── Kill Chain Tests ──────────────────────────────────────────────────────────

def test_kill_chain_honest(in_memory_db):
    """Kill chain indicators must be honest — l2_success=0 when no L2 data exists.

    With a single-layer L1 campaign and 2 test cases:
      - tc-1: decisions with acceptance_flag=1 → l1_success=1
      - tc-2: decisions with acceptance_flag=0 → l1_success=0
      - Both: l2_success=0, l3_success=0 (honest zeros, per D-09)
    """
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="killchain-test", layer="L1")

    # Two test cases
    insert_known_test_case(
        db, campaign_id, case_id="tc-kc-1", technique_id="L1_AT_01",
        payload="test A", split="harmful",
    )
    insert_known_test_case(
        db, campaign_id, case_id="tc-kc-2", technique_id="L1_AT_02",
        payload="test B", split="harmful",
    )

    # Replicas for tc-1 (2 replicas)
    rid1 = insert_known_replica(db, campaign_id, case_id="tc-kc-1", replica_num=1)
    rid2 = insert_known_replica(db, campaign_id, case_id="tc-kc-1", replica_num=2)

    # Replicas for tc-2 (1 replica)
    rid3 = insert_known_replica(db, campaign_id, case_id="tc-kc-2", replica_num=1)

    # Insert observations so compute_all has data
    insert_known_observations(db, campaign_id, [
        {"replica_id": rid1, "acceptance_flag": 1, "value": 0.9, "metric_id": "L1_ME_01"},
        {"replica_id": rid2, "acceptance_flag": 0, "value": 0.2, "metric_id": "L1_ME_01"},
        {"replica_id": rid3, "acceptance_flag": 0, "value": 0.1, "metric_id": "L1_ME_01"},
    ])

    # Insert scoring decisions
    scoring_repo = ScoringRepository(db)
    # tc-1: replica 1 gets success, replica 2 gets failure
    scoring_repo.insert_decision(rid1, ScoringDecision(
        score_value=0.9, status=ScoringStatus.COMPLETED_SUCCESS,
        technique_id="L1_AT_01", layer="L1", mode=ScoringMode.HYBRID,
    ))
    scoring_repo.insert_decision(rid2, ScoringDecision(
        score_value=0.2, status=ScoringStatus.BLOCKED,
        technique_id="L1_AT_01", layer="L1", mode=ScoringMode.HYBRID,
    ))
    # tc-2: replica 3 gets failure
    scoring_repo.insert_decision(rid3, ScoringDecision(
        score_value=0.1, status=ScoringStatus.BLOCKED,
        technique_id="L1_AT_02", layer="L1", mode=ScoringMode.HYBRID,
    ))

    # Run compute_all — this triggers _compute_kill_chains internally
    orchestrator = MetricsOrchestrator(db)
    orchestrator.compute_all(campaign_id)

    # Verify kill chain results
    kc_repo = KillChainRepository(db)
    kill_chains = kc_repo.get_kill_chains(campaign_id)
    assert len(kill_chains) == 2, (
        f"Expected 2 kill chain rows (one per test case), got {len(kill_chains)}"
    )

    # Sort by case_id for deterministic checks
    kc_by_case = {kc["case_id"]: kc for kc in kill_chains}

    # tc-1: has acceptance_flag=1 → l1_success=1
    kc1 = kc_by_case["tc-kc-1"]
    assert kc1["l1_success"] == 1, (
        f"Expected l1_success=1 for tc-1 (has success decision), got {kc1['l1_success']}"
    )
    # D-09: honest zeros for L2 and L3
    assert kc1["l2_success"] == 0, (
        f"D-09 violation: l2_success should be 0 (no L2 data), got {kc1['l2_success']}"
    )
    assert kc1["l3_success"] == 0, (
        f"D-09 violation: l3_success should be 0 (no L3 data), got {kc1['l3_success']}"
    )

    # tc-2: all acceptance_flag=0 → l1_success=0
    kc2 = kc_by_case["tc-kc-2"]
    assert kc2["l1_success"] == 0, (
        f"Expected l1_success=0 for tc-2 (no success decisions), got {kc2['l1_success']}"
    )
    # Also honest zeros for L2/L3
    assert kc2["l2_success"] == 0
    assert kc2["l3_success"] == 0


def test_kill_chain_no_fabrication(in_memory_db):
    """l2_success must NOT equal l1_success when no L2 data exists.

    Previous bug: l2_success = int(n_l1 > 0) copied L1 to L2.
    This test verifies that l2_success is 0 even when l1_success is 1.
    """
    db = in_memory_db
    campaign_id = insert_known_campaign(db, name="nofab-test", layer="L1")

    insert_known_test_case(
        db, campaign_id, case_id="tc-single", technique_id="L1_AT_01",
        payload="test", split="harmful",
    )

    # Single replica with success decision
    rid = insert_known_replica(db, campaign_id, case_id="tc-single", replica_num=1)
    insert_known_observations(db, campaign_id, [
        {"replica_id": rid, "acceptance_flag": 1, "value": 0.9, "metric_id": "L1_ME_01"},
    ])

    scoring_repo = ScoringRepository(db)
    scoring_repo.insert_decision(rid, ScoringDecision(
        score_value=0.9, status=ScoringStatus.COMPLETED_SUCCESS,
        technique_id="L1_AT_01", layer="L1", mode=ScoringMode.HYBRID,
    ))

    orchestrator = MetricsOrchestrator(db)
    orchestrator.compute_all(campaign_id)

    kc_repo = KillChainRepository(db)
    kill_chains = kc_repo.get_kill_chains(campaign_id)
    assert len(kill_chains) == 1

    kc = kill_chains[0]
    # l1_success should be 1 (we have success decisions)
    assert kc["l1_success"] == 1, (
        f"Expected l1_success=1 (success decision exists), got {kc['l1_success']}"
    )
    # D-09: l2_success must NOT equal l1_success — it must be 0
    assert kc["l2_success"] == 0, (
        f"FABRICATION BUG: l2_success={kc['l2_success']} should be 0, "
        f"not copied from l1_success={kc['l1_success']}"
    )
    assert kc["l2_success"] != kc["l1_success"], (
        f"FABRICATION BUG: l2_success ({kc['l2_success']}) should not equal "
        f"l1_success ({kc['l1_success']})"
    )
    # l3_success should also be 0
    assert kc["l3_success"] == 0


def test_far_frr_aggregates_not_equal_asr(in_memory_db):
    """Regression: FAR/FRR aggregates must use calculator value, not acceptance_flags."""
    db = in_memory_db
    from tests.conftest import insert_known_campaign, insert_known_test_case
    from tests.conftest import insert_known_replica, insert_known_observations
    from norn.metrics.orchestrator import MetricsOrchestrator
    from norn.persistence.database import MetricsRepository

    cid = insert_known_campaign(db, name="farfrr_regression", layer="L1")
    # 2 test cases: one benign, one harmful
    insert_known_test_case(db, cid, case_id="BENIGN_01", split="benign")
    insert_known_test_case(db, cid, case_id="HARMFUL_01", split="harmful")
    # 4 replicas: 2 benign, 2 harmful
    r_benign = [insert_known_replica(db, cid, case_id="BENIGN_01") for _ in range(2)]
    r_harmful = [insert_known_replica(db, cid, case_id="HARMFUL_01") for _ in range(2)]
    # Observations: benign replicas acceptance_flag=0, harmful=1 (perfect classifier)
    rows = []
    for rid in r_benign:
        rows.append({"replica_id": rid, "value": 0.1, "acceptance_flag": 0, "metric_id": "L1_ME_01", "campaign_id": cid})
    for rid in r_harmful:
        rows.append({"replica_id": rid, "value": 0.9, "acceptance_flag": 1, "metric_id": "L1_ME_01", "campaign_id": cid})
    insert_known_observations(db, cid, rows)

    orch = MetricsOrchestrator(db)
    orch.compute_all(cid)

    # Verify aggregate values are distinct
    repo = MetricsRepository(db)
    agg = repo.get_aggregates(cid)
    asr_agg = [a for a in agg if a["metric_id"] == "ASR"]
    far_agg = [a for a in agg if a["metric_id"] == "FAR"]
    frr_agg = [a for a in agg if a["metric_id"] == "FRR"]

    assert asr_agg, "ASR aggregate missing"
    assert far_agg, "FAR aggregate missing"
    assert frr_agg, "FRR aggregate missing"

    # ASR = 2/4 = 0.5 (2 harmful replicas out of 4 total accepted)
    latest_asr = asr_agg[-1]
    assert latest_asr["mean"] == 0.5, f"ASR should be 0.5, got {latest_asr['mean']}"

    # FAR = 0 (no benign replica was accepted → FP=0, TN=2)
    latest_far = far_agg[-1]
    assert latest_far["mean"] == 0.0, (
        f"FAR should be 0.0 (no false positives), got {latest_far['mean']}. "
        f"BUG: FAR aggregate may be using per-replica acceptance_flags instead of calculator result."
    )

    # FRR = 0 (no harmful replica was rejected → FN=0, TP=2)
    latest_frr = frr_agg[-1]
    assert latest_frr["mean"] == 0.0, (
        f"FRR should be 0.0 (no false negatives), got {latest_frr['mean']}. "
        f"BUG: FRR aggregate may be using per-replica acceptance_flags instead of calculator result."
    )
