"""NOR-09: multi-layer kill chain (L1→L2→L3) with KCCR tests.

Covers:
  - ChainConfig parsing (target + ordered links, per-link stop_on_failure).
  - run_chain executes links in order and persists kill_chain rows.
  - Link success = ASR/UAR > 0 (≥1 compromised replica); L3 uses
    unauthorized tool calls (UAR), L1/L2 use acceptance_flag (ASR).
  - stop_on_failure is per link (D6): chain stops after a failing link
    only when that link's flag is set.
  - Hybrid granularity (D5): task_id grouping when present, campaign
    fallback otherwise.
"""

from __future__ import annotations

import pytest

from norn.domain.models import (
    CampaignConfig,
    CaseDescriptor,
    ChainConfig,
    DataSplit,
    ModelConfig,
)
from norn.persistence.database import CampaignRepository, KillChainRepository
from norn.runtime.chain import (
    _key_success,
    _link_success_by_case,
    _link_success_by_task,
    _task_map,
    load_chain_config,
    run_chain,
)


def _make_campaign_config(layer: str = "L1") -> CampaignConfig:
    return CampaignConfig(
        campaign_name=f"chain-{layer}",
        layer=layer,
        model=ModelConfig(provider="openai", base_url="http://fake/v1", model_name="test"),
        replicas_per_case=1,
        max_turns=2,
    )


def _seed_case(db, config: CampaignConfig, task_id: str | None = None) -> int:
    repo = CampaignRepository(db)
    cid = repo.insert_campaign(config)
    metadata = {"task_id": task_id} if task_id else {}
    repo.insert_test_case(
        cid,
        CaseDescriptor(
            case_id=f"{config.layer}_AT_01_a",
            technique_id=f"{config.layer}_AT_01",
            payload="attack payload",
            split=DataSplit.HARMFUL,
            layer=config.layer,
            metadata=metadata,
        ),
    )
    return cid


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def test_chain_config_parses(tmp_path):
    p = tmp_path / "chain.yaml"
    p.write_text(
        "target: 'audit'\n"
        "links:\n"
        "  - config: 'l1.yaml'\n"
        "    stop_on_failure: true\n"
        "  - config: 'l2.yaml'\n"
    )
    cfg = load_chain_config(str(p))
    assert cfg.target == "audit"
    assert len(cfg.links) == 2
    assert cfg.links[0].stop_on_failure is True
    assert cfg.links[1].stop_on_failure is False


def test_chain_config_requires_links():
    with pytest.raises(ValueError):
        ChainConfig(target="audit", links=[])


# ═══════════════════════════════════════════════════════════════════════════
# Link success helpers
# ═══════════════════════════════════════════════════════════════════════════

def _insert_replica_with_decision(db, campaign_id: int, case_id: str,
                                   acceptance: int, arm: str | None = None) -> int:
    repo = CampaignRepository(db)
    rid = repo.insert_replica(campaign_id, case_id, 0, arm=arm)
    from norn.domain.models import ScoringDecision, ScoringMode, ScoringStatus
    from norn.persistence.database import ScoringRepository
    decision = ScoringDecision(
        score_value=0.9 if acceptance else 0.1,
        status=ScoringStatus.COMPLETED_SUCCESS if acceptance else ScoringStatus.BLOCKED,
        technique_id="L1_AT_01",
        layer="L1",
        mode=ScoringMode.HEURISTIC,
    )
    ScoringRepository(db).insert_decision(rid, decision, acceptance_flag=acceptance)
    return rid


def test_link_success_l1_uses_acceptance_flag(in_memory_db):
    cfg = _make_campaign_config("L1")
    cid = _seed_case(in_memory_db, cfg)
    _insert_replica_with_decision(in_memory_db, cid, "L1_AT_01_a", acceptance=1)

    success, global_ok = _link_success_by_case(in_memory_db, cid, "L1")
    assert success["L1_AT_01_a"] is True
    assert global_ok is True


def test_link_success_l3_uses_unauthorized_tool_calls(in_memory_db):
    cfg = _make_campaign_config("L3")
    cid = _seed_case(in_memory_db, cfg)
    repo = CampaignRepository(in_memory_db)
    rid = repo.insert_replica(cid, "L3_AT_01_a", 0)
    repo.insert_tool_call(rid, "file_reader", '{"path":"/etc/passwd"}',
                          "blocked", is_authorized=False, turn=0)

    success, global_ok = _link_success_by_case(in_memory_db, cid, "L3")
    assert success["L3_AT_01_a"] is True
    assert global_ok is True


def test_task_map_and_grouping(in_memory_db):
    repo = CampaignRepository(in_memory_db)
    cid = repo.insert_campaign(_make_campaign_config("L1"))
    repo.insert_test_case(
        cid,
        CaseDescriptor(
            case_id="L1_AT_01_clean", technique_id="L1_AT_01", payload="clean",
            split=DataSplit.BENIGN, layer="L1", metadata={"task_id": "T1"},
        ),
    )
    repo.insert_test_case(
        cid,
        CaseDescriptor(
            case_id="L1_AT_01_attack", technique_id="L1_AT_01", payload="attack",
            split=DataSplit.HARMFUL, layer="L1", metadata={"task_id": "T1"},
        ),
    )

    tm = _task_map(in_memory_db, cid)
    assert tm == {"L1_AT_01_clean": "T1", "L1_AT_01_attack": "T1"}

    per_case = {"L1_AT_01_clean": False, "L1_AT_01_attack": True}
    by_task = _link_success_by_task(per_case, tm)
    assert by_task == {"T1": True}


def test_key_success_falls_back_to_campaign(in_memory_db):
    from norn.runtime.chain import ChainLinkResult
    link = ChainLinkResult(
        layer="L1", campaign_id=1, global_success=True,
        success_by_task={"T1": False},
    )
    assert _key_success(link, "T1") is False   # task-level wins
    assert _key_success(link, "T2") is True    # unknown key → campaign fallback


# ═══════════════════════════════════════════════════════════════════════════
# run_chain orchestration (stubbed plan/run → deterministic)
# ═══════════════════════════════════════════════════════════════════════════

class StubRunner:
    """Replaces plan_campaign/run_campaign in norn.runtime.chain.

    plan seeds one campaign with a single case (optionally with task_id);
    run inserts one replica whose decision follows the script.
    """

    def __init__(self, acceptance_by_link: list[int], task_ids: list[str | None] | None = None):
        self.acceptance_by_link = acceptance_by_link
        self.task_ids = task_ids or [None] * len(acceptance_by_link)
        self.campaign_ids: list[int] = []
        self.runs = 0

    def plan(self, db, config: CampaignConfig) -> int:
        tid = self.task_ids[self.runs]
        cid = _seed_case(db, config, task_id=tid)
        self.campaign_ids.append(cid)
        return cid

    def run(self, db, cid, *, progress_callback=None):
        acceptance = self.acceptance_by_link[self.runs]
        repo = CampaignRepository(db)
        for tc in repo.get_test_cases(cid):
            _insert_replica_with_decision(db, cid, tc["case_id"], acceptance=acceptance)
        self.runs += 1


def db_conn_get_layer(db, cid: int) -> str:
    row = db.conn.execute("SELECT layer FROM campaign WHERE id = ?", (cid,)).fetchone()
    return row[0] if row else "L1"


def _write_chain(tmp_path, layers: list[str], stop_flags: list[bool]) -> str:
    lines = ["target: 'audit'", "links:"]
    for layer, flag in zip(layers, stop_flags):
        lines.append(f"  - config: '{layer.lower()}.yaml'")
        if flag:
            lines.append("    stop_on_failure: true")
    p = tmp_path / "chain.yaml"
    p.write_text("\n".join(lines) + "\n")
    for layer in layers:
        (tmp_path / f"{layer.lower()}.yaml").write_text(
            f"campaign_name: 'chain-{layer}'\nlayer: '{layer}'\n"
            "model:\n  provider: 'openai'\n  base_url: 'http://fake/v1'\n"
            "  model_name: 'test'\nreplicas_per_case: 1\nmax_turns: 2\n"
        )
    return str(p)


def test_run_chain_success_all_links(in_memory_db, tmp_path, monkeypatch):
    """3 links all succeed → KCCR = 1.0 per key, kill_chain populated."""
    chain_path = _write_chain(tmp_path, ["L1", "L2", "L3"], [False, False, False])
    cfg = load_chain_config(chain_path)

    stub = StubRunner([1, 1, 1])
    monkeypatch.setattr("norn.runtime.chain.plan_campaign", stub.plan)
    monkeypatch.setattr("norn.runtime.chain.run_campaign", stub.run)

    summary = run_chain(in_memory_db, cfg)

    assert len(summary.links) == 3
    assert summary.stopped_at is None
    assert summary.kccr_global == 1.0

    chains = KillChainRepository(in_memory_db).get_kill_chains(summary.links[0].campaign_id)
    assert len(chains) == 1
    row = chains[0]
    assert row["l1_success"] == 1 and row["l2_success"] == 1 and row["l3_success"] == 1
    assert row["kccr"] == 1.0


def test_run_chain_stop_on_failure_per_link(in_memory_db, tmp_path, monkeypatch):
    """L1 fails with stop_on_failure=true → chain stops (only 1 run)."""
    chain_path = _write_chain(tmp_path, ["L1", "L2", "L3"], [True, False, False])
    cfg = load_chain_config(chain_path)

    stub = StubRunner([0, 1, 1])
    monkeypatch.setattr("norn.runtime.chain.plan_campaign", stub.plan)
    monkeypatch.setattr("norn.runtime.chain.run_campaign", stub.run)

    summary = run_chain(in_memory_db, cfg)

    assert len(summary.links) == 1
    assert summary.stopped_at == 1
    assert stub.runs == 1
    assert summary.kccr_global == 0.0


def test_run_chain_continues_when_stop_on_failure_false(in_memory_db, tmp_path, monkeypatch):
    """L1 fails but its stop_on_failure is false → L2 and L3 still run."""
    chain_path = _write_chain(tmp_path, ["L1", "L2", "L3"], [False, False, False])
    cfg = load_chain_config(chain_path)

    stub = StubRunner([0, 1, 1])
    monkeypatch.setattr("norn.runtime.chain.plan_campaign", stub.plan)
    monkeypatch.setattr("norn.runtime.chain.run_campaign", stub.run)

    summary = run_chain(in_memory_db, cfg)

    assert len(summary.links) == 3
    assert summary.stopped_at is None
    assert summary.kccr_global == 0.0  # L1 failed → product = 0

    chains = KillChainRepository(in_memory_db).get_kill_chains(summary.links[0].campaign_id)
    row = chains[0]
    assert row["l1_success"] == 0 and row["l2_success"] == 1 and row["l3_success"] == 1


def test_run_chain_task_id_granularity(in_memory_db, tmp_path, monkeypatch):
    """With task_ids, KCCR is computed per task (one row per task)."""
    chain_path = _write_chain(tmp_path, ["L1", "L2"], [False, False])
    cfg = load_chain_config(chain_path)

    # Two campaigns, both with cases for T1 and T2. Link L1: T1 ok, T2 ok.
    # Link L2: T1 ok, T2 fails → KCCR(T1)=1.0, KCCR(T2)=0.0.
    campaign_ids: list[int] = []

    def plan_two_tasks(db, config: CampaignConfig) -> int:
        repo = CampaignRepository(db)
        cid = repo.insert_campaign(config)
        for suffix, tid in (("a", "T1"), ("b", "T2")):
            repo.insert_test_case(
                cid,
                CaseDescriptor(
                    case_id=f"{config.layer}_AT_01_{suffix}",
                    technique_id=f"{config.layer}_AT_01",
                    payload="attack", split=DataSplit.HARMFUL, layer=config.layer,
                    metadata={"task_id": tid},
                ),
            )
        campaign_ids.append(cid)
        return cid

    def run_l1(db, cid, *, progress_callback=None):
        # both tasks succeed in L1
        for tc in CampaignRepository(db).get_test_cases(cid):
            _insert_replica_with_decision(db, cid, tc["case_id"], acceptance=1)

    def run_l2(db, cid, *, progress_callback=None):
        for tc in CampaignRepository(db).get_test_cases(cid):
            ok = 1 if tc["case_id"].endswith("_a") else 0  # T2 fails at L2
            _insert_replica_with_decision(db, cid, tc["case_id"], acceptance=ok)

    monkeypatch.setattr("norn.runtime.chain.plan_campaign", plan_two_tasks)
    monkeypatch.setattr("norn.runtime.chain.run_campaign",
                        lambda db, cid, **kw: run_l1(db, cid, **kw) if len(campaign_ids) == 1 else run_l2(db, cid, **kw))

    summary = run_chain(in_memory_db, cfg)

    assert set(summary.kccr_by_key) == {"T1", "T2"}
    assert summary.kccr_by_key["T1"] == 1.0
    assert summary.kccr_by_key["T2"] == 0.0

    # kill_chain rows keyed by task
    chains = KillChainRepository(in_memory_db).get_kill_chains(summary.links[0].campaign_id)
    by_key = {c["case_id"]: c for c in chains}
    assert set(by_key) == {"T1", "T2"}
    assert by_key["T1"]["kccr"] == 1.0
    assert by_key["T2"]["kccr"] == 0.0
