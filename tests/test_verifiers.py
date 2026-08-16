"""NOR-25: pluggable verifiers — ScoringConfig.verifiers + registry + ensemble.

Covers:
  - effective_verifiers() / effective_vote_strategy() (decisions D1/D2):
    signals by default on new configs, legacy mode translation preserved,
    judge_arbiter by default (honest status), explicit values always win.
  - fail-fast validation of unknown verifier names (model + build_scorer).
  - plan_campaign persists the RESOLVED pipeline so the DB round-trip
    cannot defeat D1/D2.
  - build_scorer resolves verifier lists (single voter returns the voter
    directly; heuristic_legacy+judge keeps HybridScorer; other combos →
    EnsembleScorer).
  - EnsembleScorer combination strategies (majority 2-voter = legacy
    hybrid semantics, N-voter majority, veto, reasoning H/L format).
"""
from __future__ import annotations

import json

import pytest

from norn.domain.models import (
    KNOWN_VERIFIERS,
    CampaignConfig,
    ScoringConfig,
    ScoringMode,
    ScoringStatus,
    VoteStrategy,
)
from norn.runtime.campaign import _campaign_config_from_db, plan_campaign
from norn.scoring.scorers import (
    _VERIFIER_REGISTRY,
    EnsembleScorer,
    HeuristicScorer,
    HybridScorer,
    LLMJudgeScorer,
    build_scorer,
)
from tests.conftest import insert_known_campaign


class _FakeHeuristic(HeuristicScorer):
    """Deterministic scripted voter for ensemble tests."""

    name = "heuristic_legacy"

    def __init__(self, status: ScoringStatus, score: float):
        self._status = status
        self._score = score

    def score_response(self, prompt, response, technique_id, layer, context=None,
                       split=None, replica_id=None):
        from norn.domain.models import ScoringDecision
        return ScoringDecision(
            score_value=self._score, status=self._status,
            technique_id=technique_id, layer=layer, mode=ScoringMode.HEURISTIC,
            reasoning=f"fake H {self._status.value}",
        )

    def supports_technique(self, technique_id: str) -> bool:
        return True


class _FakeJudge:
    name = "judge"

    def __init__(self, status: ScoringStatus, score: float):
        self._status = status
        self._score = score

    def score_response(self, prompt, response, technique_id, layer, context=None,
                       split=None, replica_id=None):
        from norn.domain.models import ScoringDecision
        return ScoringDecision(
            score_value=self._score, status=self._status,
            technique_id=technique_id, layer=layer, mode=ScoringMode.LLM_JUDGE,
            reasoning=f"fake L {self._status.value}",
        )

    def supports_technique(self, technique_id: str) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Registry / known names
# ═══════════════════════════════════════════════════════════════════════════

def test_registry_matches_known_verifiers():
    """The registry and ScoringConfig.KNOWN_VERIFIERS must stay in sync."""
    assert set(_VERIFIER_REGISTRY) == set(KNOWN_VERIFIERS)


# ═══════════════════════════════════════════════════════════════════════════
# ScoringConfig.effective_verifiers (D2)
# ═══════════════════════════════════════════════════════════════════════════

def test_effective_verifiers_default_is_signals_plus_judge():
    """D2: configs without `mode` get the new signals pipeline by default."""
    assert ScoringConfig().effective_verifiers() == ["heuristic_signals", "judge"]


def test_effective_verifiers_explicit_mode_keeps_legacy_verifier():
    """D2: an explicit legacy `mode` keeps its exact verifier."""
    assert ScoringConfig(mode="heuristic").effective_verifiers() == ["heuristic_legacy"]
    assert ScoringConfig(mode="llm_judge").effective_verifiers() == ["judge"]
    assert ScoringConfig(mode="hybrid").effective_verifiers() == ["heuristic_legacy", "judge"]


def test_effective_verifiers_explicit_verifiers_win_over_mode():
    assert ScoringConfig(mode="heuristic", verifiers=["judge"]).effective_verifiers() == ["judge"]
    assert ScoringConfig(verifiers=["heuristic_legacy"]).effective_verifiers() == ["heuristic_legacy"]


def test_unknown_verifier_fails_fast_in_model():
    with pytest.raises(ValueError, match="Unknown verifier"):
        ScoringConfig(verifiers=["bogus_scorer"])


# ═══════════════════════════════════════════════════════════════════════════
# ScoringConfig.effective_vote_strategy (D1)
# ═══════════════════════════════════════════════════════════════════════════

def test_effective_vote_strategy_default_is_judge_arbiter():
    """D1: 'respuesta honesta siempre' — the judge arbitrates by default."""
    assert ScoringConfig().effective_vote_strategy() == VoteStrategy.JUDGE_ARBITER
    assert ScoringConfig(mode="hybrid").effective_vote_strategy() == VoteStrategy.JUDGE_ARBITER
    assert ScoringConfig(verifiers=["heuristic_legacy", "judge"]).effective_vote_strategy() == VoteStrategy.JUDGE_ARBITER


def test_effective_vote_strategy_explicit_wins():
    assert ScoringConfig(vote_strategy="veto").effective_vote_strategy() == VoteStrategy.VETO
    assert ScoringConfig(mode="hybrid", vote_strategy="majority").effective_vote_strategy() == VoteStrategy.MAJORITY


# ═══════════════════════════════════════════════════════════════════════════
# plan_campaign persists the resolved pipeline (DB round-trip)
# ═══════════════════════════════════════════════════════════════════════════

def _campaign_config(**scoring_kwargs) -> CampaignConfig:
    return CampaignConfig(
        campaign_name="t",
        layer="L1",
        scoring=ScoringConfig(**scoring_kwargs),
        replicas_per_case=1,
    )


def test_plan_persists_resolved_default_pipeline(in_memory_db):
    """A config without `mode` is stored with signals+judge / judge_arbiter,
    so the run phase sees the same pipeline after the round-trip."""
    cid = plan_campaign(in_memory_db, _campaign_config())
    stored = json.loads(in_memory_db.conn.execute(
        "SELECT config_json FROM campaign WHERE id=?", (cid,)
    ).fetchone()[0])
    assert stored["scoring"]["verifiers"] == ["heuristic_signals", "judge"]
    assert stored["scoring"]["vote_strategy"] == "judge_arbiter"
    # round-trip keeps the effective pipeline
    rebuilt = _campaign_config_from_db(in_memory_db, cid)
    assert rebuilt.scoring.effective_verifiers() == ["heuristic_signals", "judge"]
    assert rebuilt.scoring.effective_vote_strategy() == VoteStrategy.JUDGE_ARBITER


def test_plan_persists_explicit_mode_translation(in_memory_db):
    """Explicit `mode: hybrid` is stored as heuristic_legacy+judge with the
    judge_arbiter strategy (D1), and explicit strategy survives."""
    cid = plan_campaign(in_memory_db, _campaign_config(mode="hybrid", vote_strategy="majority"))
    rebuilt = _campaign_config_from_db(in_memory_db, cid)
    assert rebuilt.scoring.effective_verifiers() == ["heuristic_legacy", "judge"]
    assert rebuilt.scoring.effective_vote_strategy() == VoteStrategy.MAJORITY


def test_legacy_stored_config_without_verifiers_keeps_behavior(in_memory_db):
    """Old full-dump configs (no `verifiers` field) keep the legacy pipeline:
    mode hybrid + stored majority → heuristic_legacy + judge + majority."""
    cid = insert_known_campaign(in_memory_db, name="old", layer="L1")
    conn = in_memory_db.conn
    conn.execute(
        "UPDATE campaign SET config_json=? WHERE id=?",
        (json.dumps({"campaign_name": "old", "layer": "L1", "scoring": {"mode": "hybrid", "vote_strategy": "majority"}}), cid),
    )
    conn.commit()
    rebuilt = _campaign_config_from_db(in_memory_db, cid)
    assert rebuilt.scoring.effective_verifiers() == ["heuristic_legacy", "judge"]
    assert rebuilt.scoring.effective_vote_strategy() == VoteStrategy.MAJORITY


# ═══════════════════════════════════════════════════════════════════════════
# build_scorer with verifier lists
# ═══════════════════════════════════════════════════════════════════════════

def test_build_scorer_single_verifier_returns_voter_directly(monkeypatch):
    monkeypatch.setattr("norn.scoring.scorers.build_provider", lambda name: None)
    assert isinstance(build_scorer("hybrid", verifiers=["heuristic_legacy"]), HeuristicScorer)
    assert isinstance(build_scorer("hybrid", verifiers=["judge"], judge_model="j"), LLMJudgeScorer)


def test_build_scorer_classic_hybrid_keeps_hybrid_scorer(monkeypatch):
    monkeypatch.setattr("norn.scoring.scorers.build_provider", lambda name: None)
    scorer = build_scorer("hybrid", verifiers=["heuristic_legacy", "judge"], vote_strategy="majority")
    assert isinstance(scorer, HybridScorer)
    assert isinstance(scorer.heuristic, HeuristicScorer)
    assert isinstance(scorer.llm_judge, LLMJudgeScorer)


def test_build_scorer_signals_plus_judge_is_ensemble(monkeypatch):
    monkeypatch.setattr("norn.scoring.scorers.build_provider", lambda name: None)
    scorer = build_scorer("hybrid", verifiers=["heuristic_signals", "judge"], vote_strategy="judge_arbiter")
    assert isinstance(scorer, EnsembleScorer)
    assert not isinstance(scorer, HybridScorer)
    assert scorer.voter_names == ["heuristic_signals", "judge"]


def test_build_scorer_unknown_verifier_raises():
    with pytest.raises(ValueError, match="Unknown verifier"):
        build_scorer("hybrid", verifiers=["nope"])


def test_build_scorer_mode_translation_without_verifiers(monkeypatch):
    """verifiers=None → legacy translation (retrocompat)."""
    monkeypatch.setattr("norn.scoring.scorers.build_provider", lambda name: None)
    assert isinstance(build_scorer("heuristic"), HeuristicScorer)
    assert isinstance(build_scorer("llm_judge", judge_model="j"), LLMJudgeScorer)
    assert isinstance(build_scorer("hybrid"), HybridScorer)


# ═══════════════════════════════════════════════════════════════════════════
# EnsembleScorer strategies
# ═══════════════════════════════════════════════════════════════════════════

def _decide(ensemble, *args, **kwargs):
    return ensemble.score_response("p", "r", "L1_AT_01", "L1", *args, **kwargs)


def test_ensemble_majority_two_voters_matches_legacy_hybrid():
    """`verifiers=[heuristic_legacy, judge]` + majority ≡ mode: hybrid."""
    legacy = HybridScorer(_FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
                          _FakeJudge(ScoringStatus.BLOCKED, 0.1), vote_strategy="majority")
    ensemble = EnsembleScorer([_FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
                               _FakeJudge(ScoringStatus.BLOCKED, 0.1)],
                              vote_strategy="majority",
                              voter_names=["heuristic_legacy", "judge"])
    d_legacy = _decide(legacy)
    d_ensemble = _decide(ensemble)
    assert d_ensemble.status == d_legacy.status == ScoringStatus.COMPLETED_SUCCESS
    assert d_ensemble.score_value == d_legacy.score_value == 0.5
    assert d_ensemble.reasoning == d_legacy.reasoning == "H(completed_success=0.90) L(blocked=0.10)"
    assert d_ensemble.votes == d_legacy.votes == [("heuristic_legacy", 0.9), ("judge", 0.1)]


def test_ensemble_majority_all_ambiguous():
    ensemble = EnsembleScorer([_FakeHeuristic(ScoringStatus.AMBIGUOUS, 0.5),
                               _FakeJudge(ScoringStatus.AMBIGUOUS, 0.5)],
                              vote_strategy="majority")
    d = _decide(ensemble)
    assert d.status == ScoringStatus.AMBIGUOUS


def test_ensemble_veto_blocks_when_any_voter_blocks():
    ensemble = EnsembleScorer([_FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
                               _FakeJudge(ScoringStatus.BLOCKED, 0.1)],
                              vote_strategy="veto")
    d = _decide(ensemble)
    assert d.status == ScoringStatus.BLOCKED
    assert d.score_value == 0.1
    assert "Veto:" in d.reasoning


def test_ensemble_n_voter_majority_most_common_status():
    voters = [
        _FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
        _FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.8),
        _FakeJudge(ScoringStatus.AMBIGUOUS, 0.5),
    ]
    ensemble = EnsembleScorer(voters, vote_strategy="majority",
                              voter_names=["heuristic_signals", "heuristic_legacy", "judge"])
    d = _decide(ensemble)
    assert d.status == ScoringStatus.COMPLETED_SUCCESS
    assert d.score_value == pytest.approx((0.9 + 0.8 + 0.5) / 3)


def test_ensemble_n_voter_majority_tie_is_ambiguous():
    voters = [
        _FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
        _FakeHeuristic(ScoringStatus.BLOCKED, 0.1),
        _FakeJudge(ScoringStatus.AMBIGUOUS, 0.5),
    ]
    ensemble = EnsembleScorer(voters, vote_strategy="majority")
    d = _decide(ensemble)
    assert d.status == ScoringStatus.AMBIGUOUS


def test_ensemble_reasoning_keeps_h_l_format():
    """The H(...) L(...) format survives for the live -v regex and the E2E
    verification scripts."""
    ensemble = EnsembleScorer([_FakeHeuristic(ScoringStatus.COMPLETED_SUCCESS, 0.9),
                               _FakeJudge(ScoringStatus.BLOCKED, 0.1)],
                              vote_strategy="majority",
                              voter_names=["heuristic_signals", "judge"])
    d = _decide(ensemble)
    assert d.reasoning == "H(completed_success=0.90) L(blocked=0.10)"
