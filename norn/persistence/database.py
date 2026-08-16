"""SQLite persistence layer for campaign data.

Implements the multi-domain data model described in chapter 3.6.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from norn.domain.models import (
    CampaignConfig,
    CampaignState,
    CaseDescriptor,
    ScoringDecision,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """Manages the SQLite connection and schema."""

    def __init__(self, db_path: str = "norn.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.close()


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def init_schema(db: Database):
    """Create all tables per the multi-domain schema (via migrations)."""
    db.connect()
    migrate(db)


def _migration_files() -> list[tuple[int, Path]]:
    """Return (version, path) pairs for migration scripts, sorted by version."""
    files: list[tuple[int, Path]] = []
    for path in MIGRATIONS_DIR.glob("*.sql"):
        name = path.name.split("_", 1)[0]
        try:
            version = int(name)
        except ValueError:
            raise ValueError(
                f"Invalid migration filename (expected NNN_*.sql): {path.name}"
            ) from None
        files.append((version, path))
    return sorted(files, key=lambda item: item[0])


def current_version(db: Database) -> int:
    """Return the schema version stored in PRAGMA user_version."""
    conn = db.conn or db.connect()
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(db: Database) -> int:
    """Apply pending migrations in order and return the new schema version.

    Idempotent: only migrations with version > current are applied. The base
    script (001) uses CREATE TABLE IF NOT EXISTS so databases created before
    schema versioning (user_version=0) migrate without friction. Migrations
    that only add a column tolerate the "duplicate column name" error so
    databases that already received the column via the old inline ALTER keep
    working.
    """
    conn = db.conn or db.connect()
    start = current_version(db)
    for version, path in _migration_files():
        if version <= start:
            continue
        try:
            conn.executescript(path.read_text(encoding="utf-8"))
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc):
                raise
            # Column already present (pre-versioned db) — migration is a no-op.
        # PRAGMA does not support parameter binding; version is an int parsed
        # from our own migration filenames (NNN_*.sql), never user input.
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
    return current_version(db)


def seed_catalog(db: Database):
    """Populate the catalog tables with taxonomy data."""
    from norn.domain.taxonomy import (
        ATTACK_TECHNIQUES,
        LAYER_CATALOG,
        METRIC_DEFINITIONS,
        TECHNIQUE_MAP,
    )

    conn = db.conn or db.connect()

    for lid, info in LAYER_CATALOG.items():
        conn.execute(
            "INSERT OR IGNORE INTO layer_catalog (id, name, description) VALUES (?, ?, ?)",
            (lid, info["name"], info["description"]),
        )

    for tid, tech in ATTACK_TECHNIQUES.items():
        conn.execute(
            "INSERT OR IGNORE INTO attack_technique (id, name, layer_id, description, owasp_tags, mitre_tags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tech.id, tech.name, tech.layer, tech.description,
             ",".join(tech.owasp), ",".join(tech.mitre_atlas)),
        )

    for mid, mdef in METRIC_DEFINITIONS.items():
        conn.execute(
            "INSERT OR IGNORE INTO metric_definition (id, name, layer_id, formula, direction, unit, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mdef.id, mdef.name, mdef.layer, mdef.formula, mdef.direction, mdef.unit, mdef.description),
        )

    for tid, mapping in TECHNIQUE_MAP.items():
        if "owasp" in mapping:
            conn.execute(
                "INSERT OR IGNORE INTO framework_mapping (target_type, target_id, framework, framework_id, relation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                ("technique", tid, "OWASP_LLM_TOP10", mapping["owasp"], "direct"),
            )
        if "mitre" in mapping:
            conn.execute(
                "INSERT OR IGNORE INTO framework_mapping (target_type, target_id, framework, framework_id, relation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                ("technique", tid, "MITRE_ATLAS", mapping["mitre"], "direct"),
            )

    conn.commit()


# ── Repositories ─────────────────────────────────────────────────────────────

class BaseRepository:
    def __init__(self, db: Database):
        self.db = db

    @property
    def conn(self) -> sqlite3.Connection:
        return self.db.conn or self.db.connect()


def _scope_clause(arm: str | None, temperature: float | None) -> tuple[str, list]:
    """Build (SQL fragment, params) for arm/temperature scope filters.

    NOR-08 arms and NOR-21 temperature sweeps share the same per-replica
    filtering pattern (run_replica joined as ``rr``). Both filters can be
    combined (arms × temps) — the caller decides which scopes to compute.
    """
    clauses: list[str] = []
    params: list = []
    if arm is not None:
        clauses.append("rr.arm = ?")
        params.append(arm)
    if temperature is not None:
        clauses.append("rr.temperature = ?")
        params.append(temperature)
    suffix = (" AND " + " AND ".join(clauses)) if clauses else ""
    return suffix, params


class CampaignRepository(BaseRepository):
    """Repository for campaign, test cases, replicas, and results."""

    def insert_campaign(self, config: CampaignConfig) -> int:
        cur = self.conn.execute(
            "INSERT INTO campaign (name, layer, state, description, config_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (config.campaign_name, config.layer, CampaignState.PLANNED.value,
             config.description, config.model_dump_json(), _now(), _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_campaign(self, campaign_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM campaign WHERE id = ?", (campaign_id,)).fetchone()
        return dict(row) if row else None

    def list_campaigns(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT id, name, layer, state, description, created_at FROM campaign ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def update_state(self, campaign_id: int, state: CampaignState):
        self.conn.execute(
            "UPDATE campaign SET state = ?, updated_at = ? WHERE id = ?",
            (state.value, _now(), campaign_id),
        )
        self.conn.commit()

    def insert_test_case(self, campaign_id: int, case: CaseDescriptor) -> int:
        import hashlib
        ph = hashlib.sha256(case.payload.encode()).hexdigest()[:16]
        cur = self.conn.execute(
            "INSERT INTO test_case (campaign_id, case_id, technique_id, payload, split, metadata_json, payload_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (campaign_id, case.case_id, case.technique_id, case.payload,
             case.split.value, json.dumps(case.metadata), ph),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_replica(self, campaign_id: int, case_id: str, replica: int,
                       temperature: float = 0.0, top_p: float = 0.9, seed: int | None = 42,
                       error_message: str | None = None, arm: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO run_replica (campaign_id, case_id, replica, state, temperature, top_p, seed, error_message, arm, created_at) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
            (campaign_id, case_id, replica, temperature, top_p, seed, error_message, arm, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_replica_state(self, replica_id: int, state: str,
                             error_message: str | None = None):
        self.conn.execute(
            "UPDATE run_replica SET state = ?, error_message = COALESCE(?, error_message) WHERE id = ?",
            (state, error_message, replica_id),
        )
        self.conn.commit()

    def insert_turn_event(self, replica_id: int, turn: int, prompt: str, response: str,
                          tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0,
                          role: str = "user", model: str | None = None) -> int:
        """Insert a turn event.

        ``role`` defaults to ``user`` for the audited model's turns; the
        LLM judge records its calls with ``role='judge'`` (NOR-07) so cost
        estimation can split model vs judge tokens and conversation exports
        can filter judge verdicts out. ``model`` (NOR-19) names the judge
        model that produced the call so multi-model ensembles keep
        per-model cost attribution.
        """
        cur = self.conn.execute(
            "INSERT INTO turn_event (replica_id, turn, prompt, response, role, tokens_in, tokens_out, latency_ms, model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (replica_id, turn, prompt, response, role, tokens_in, tokens_out, latency_ms, model),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_tool_call(self, replica_id: int, tool_name: str, tool_params: str,
                         tool_result: str, is_authorized: bool, turn: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO tool_call_event (replica_id, tool_name, tool_params, tool_result, is_authorized, turn) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (replica_id, tool_name, tool_params, tool_result, int(is_authorized), turn),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_replicas(self, campaign_id: int, arm: str | None = None,
                     temperature: float | None = None) -> list[dict[str, Any]]:
        suffix, params = _scope_clause(arm, temperature)
        rows = self.conn.execute(
            f"SELECT rr.* FROM run_replica rr WHERE rr.campaign_id = ?{suffix} ORDER BY rr.id",
            [campaign_id, *params],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_test_cases(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM test_case WHERE campaign_id = ? ORDER BY id", (campaign_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_events(self, replica_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM turn_event WHERE replica_id = ? ORDER BY turn", (replica_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_calls(self, campaign_id: int, arm: str | None = None,
                       temperature: float | None = None) -> list[dict[str, Any]]:
        suffix, params = _scope_clause(arm, temperature)
        rows = self.conn.execute(
            "SELECT tce.* FROM tool_call_event tce "
            "JOIN run_replica rr ON tce.replica_id = rr.id "
            f"WHERE rr.campaign_id = ?{suffix} ORDER BY tce.id",
            [campaign_id, *params],
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_retrieval_event(self, replica_id: int, poisoned: bool,
                               top_k: int, retrieved: list) -> int:
        cur = self.conn.execute(
            "INSERT INTO retrieval_event (replica_id, poisoned_retrieval, top_k, retrieved_json) "
            "VALUES (?, ?, ?, ?)",
            (replica_id, int(poisoned), top_k, json.dumps(retrieved)),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_retrieval_events(self, campaign_id: int, arm: str | None = None,
                             temperature: float | None = None) -> list[dict[str, Any]]:
        suffix, params = _scope_clause(arm, temperature)
        rows = self.conn.execute(
            "SELECT re.* FROM retrieval_event re "
            "JOIN run_replica rr ON re.replica_id = rr.id "
            f"WHERE rr.campaign_id = ?{suffix} ORDER BY re.id",
            [campaign_id, *params],
        ).fetchall()
        return [dict(r) for r in rows]


class MetricsRepository(BaseRepository):
    """Repository for metric observations and aggregates."""

    def insert_observation(self, campaign_id: int, metric_id: str, replica_id: int | None,
                           value: float, confidence: float = 1.0, acceptance_flag: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO metric_observation (campaign_id, metric_id, replica_id, value, confidence, acceptance_flag) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, metric_id, replica_id, value, confidence, acceptance_flag),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_observations(self, campaign_id: int, metric_id: str | None = None,
                         arm: str | None = None,
                         temperature: float | None = None) -> list[dict[str, Any]]:
        """Return per-replica observations, optionally filtered by arm (NOR-08)
        or temperature (NOR-21)."""
        suffix, params = _scope_clause(arm, temperature)
        if metric_id:
            rows = self.conn.execute(
                "SELECT mo.* FROM metric_observation mo "
                "JOIN run_replica rr ON mo.replica_id = rr.id "
                f"WHERE rr.campaign_id = ? AND mo.metric_id = ?{suffix} "
                "AND mo.replica_id IS NOT NULL",
                [campaign_id, metric_id, *params],
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT mo.* FROM metric_observation mo "
                "JOIN run_replica rr ON mo.replica_id = rr.id "
                f"WHERE rr.campaign_id = ?{suffix} AND mo.replica_id IS NOT NULL",
                [campaign_id, *params],
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_aggregate(self, campaign_id: int, metric_id: str, scope_type: str,
                         mean: float, std_dev: float, ci95_lower: float, ci95_upper: float,
                         median: float, min_val: float, max_val: float, total: int) -> int:
        cur = self.conn.execute(
            "INSERT INTO metric_aggregate (campaign_id, metric_id, scope_type, mean, std_dev, "
            "ci95_lower, ci95_upper, median, min_val, max_val, total_observations) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (campaign_id, metric_id, scope_type, mean, std_dev, ci95_lower, ci95_upper,
             median, min_val, max_val, total),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_aggregates(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM metric_aggregate WHERE campaign_id = ? ORDER BY metric_id",
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_aggregates(self, campaign_id: int, metric_id: str, scope_type: str = "campaign") -> None:
        self.conn.execute(
            "DELETE FROM metric_aggregate WHERE campaign_id = ? AND metric_id = ? AND scope_type = ?",
            (campaign_id, metric_id, scope_type),
        )
        self.conn.commit()


class ScoringRepository(BaseRepository):
    """Repository for scoring decisions and votes."""

    def insert_decision(self, replica_id: int, decision: ScoringDecision,
                        acceptance_flag: int | None = None) -> int:
        if acceptance_flag is None:
            acceptance_flag = 1 if decision.status.value == "completed_success" else 0
        cur = self.conn.execute(
            "INSERT INTO scoring_decision (replica_id, technique_id, score_value, status, mode, reasoning, acceptance_flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (replica_id, decision.technique_id, decision.score_value,
             decision.status.value, decision.mode.value, decision.reasoning, acceptance_flag),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_vote(self, decision_id: int, voter_type: str, vote: float, reasoning: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO scoring_vote (decision_id, voter_type, vote, reasoning) VALUES (?, ?, ?, ?)",
            (decision_id, voter_type, vote, reasoning),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_decisions(self, campaign_id: int, arm: str | None = None,
                      temperature: float | None = None) -> list[dict[str, Any]]:
        suffix, params = _scope_clause(arm, temperature)
        rows = self.conn.execute(
            "SELECT sd.* FROM scoring_decision sd "
            "JOIN run_replica rr ON sd.replica_id = rr.id "
            f"WHERE rr.campaign_id = ?{suffix} ORDER BY sd.id",
            [campaign_id, *params],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_votes(self, campaign_id: int, arm: str | None = None,
                  temperature: float | None = None) -> list[dict[str, Any]]:
        """Return scoring votes per replica for a campaign.

        Each row: {replica_id, voter_type, vote, reasoning}. Used by the
        metrics orchestrator to recover the judge's individual verdict
        (compromise ground truth) for FAR/FRR. Optionally filtered by arm
        (NOR-08) or temperature (NOR-21).
        """
        suffix, params = _scope_clause(arm, temperature)
        rows = self.conn.execute(
            "SELECT rr.id AS replica_id, sv.voter_type, sv.vote, sv.reasoning "
            "FROM scoring_vote sv "
            "JOIN scoring_decision sd ON sv.decision_id = sd.id "
            "JOIN run_replica rr ON sd.replica_id = rr.id "
            f"WHERE rr.campaign_id = ?{suffix} ORDER BY sv.id",
            [campaign_id, *params],
        ).fetchall()
        return [dict(r) for r in rows]


class KillChainRepository(BaseRepository):
    """Repository for kill-chain results and risk assessments."""

    def insert_kill_chain(self, campaign_id: int, case_id: str,
                          l1_success: int, l2_success: int, l3_success: int, kccr: float) -> int:
        cur = self.conn.execute(
            "INSERT INTO kill_chain_result (campaign_id, case_id, l1_success, l2_success, l3_success, kccr) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, case_id, l1_success, l2_success, l3_success, kccr),
        )
        self.conn.commit()
        return cur.lastrowid

    def upsert_kill_chain(self, campaign_id: int, case_id: str,
                          l1_success: int, l2_success: int, l3_success: int, kccr: float) -> None:
        """Insert or replace a kill-chain row (NOR-09 chains rewrite per case)."""
        self.conn.execute(
            "DELETE FROM kill_chain_result WHERE campaign_id = ? AND case_id = ?",
            (campaign_id, case_id),
        )
        self.insert_kill_chain(campaign_id, case_id, l1_success, l2_success, l3_success, kccr)

    def get_kill_chains(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM kill_chain_result WHERE campaign_id = ?", (campaign_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_risk(self, campaign_id: int, case_id: str,
                    exploitation: float, impact: float, stealth: float,
                    w_e: float = 0.4, w_i: float = 0.4, w_s: float = 0.2) -> int:
        severity = "low"
        weighted = exploitation * w_e + impact * w_i + stealth * w_s
        if weighted >= 0.7:
            severity = "critical"
        elif weighted >= 0.5:
            severity = "high"
        elif weighted >= 0.3:
            severity = "medium"
        cur = self.conn.execute(
            "INSERT INTO risk_assessment (campaign_id, case_id, exploitation_score, impact_score, "
            "stealth_score, weight_e, weight_i, weight_s, severity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (campaign_id, case_id, exploitation, impact, stealth, w_e, w_i, w_s, severity),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_risks(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM risk_assessment WHERE campaign_id = ?", (campaign_id,)
        ).fetchall()
        return [dict(r) for r in rows]


class ArtifactRepository(BaseRepository):
    """Repository for artifacts and audit events."""

    def insert_artifact(self, campaign_id: int, name: str, path: str, fmt: str,
                        sha256: str = "", size_bytes: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO artifact (campaign_id, name, path, format, sha256, size_bytes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, name, path, fmt, sha256, size_bytes),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_audit(self, campaign_id: int, event_type: str, target_entity: str,
                     old_value: str = "", new_value: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO audit_event (campaign_id, event_type, target_entity, old_value, new_value, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (campaign_id, event_type, target_entity, old_value, new_value, _now()),
        )
        self.conn.commit()
        return cur.lastrowid


class CostRepository(BaseRepository):
    """Repository for the model price catalog (NOR-07)."""

    def upsert_model_cost(self, model: str, provider: str, input_per_1k: float,
                          output_per_1k: float, currency: str = "USD",
                          source: str | None = None) -> int:
        """Insert or update a price row for (model, provider)."""
        cur = self.conn.execute(
            "INSERT INTO model_cost (model, provider, input_per_1k, output_per_1k, "
            "currency, source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model, provider) DO UPDATE SET "
            "input_per_1k = excluded.input_per_1k, "
            "output_per_1k = excluded.output_per_1k, "
            "currency = excluded.currency, "
            "source = excluded.source, "
            "updated_at = excluded.updated_at",
            (model, provider, input_per_1k, output_per_1k, currency, source, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_model_cost(self, model: str, provider: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM model_cost WHERE model = ? AND provider = ?",
            (model, provider),
        ).fetchone()
        return dict(row) if row else None

    def get_all_model_costs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM model_cost ORDER BY provider, model"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_tokens(self, campaign_id: int) -> list[dict[str, Any]]:
        """Token usage per turn event of a campaign (with role and model)."""
        rows = self.conn.execute(
            "SELECT te.role, te.tokens_in, te.tokens_out, te.model "
            "FROM turn_event te "
            "JOIN run_replica rr ON te.replica_id = rr.id "
            "WHERE rr.campaign_id = ?",
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]


class CampaignDataCollector:
    """Collects all campaign data for export."""

    def __init__(self, db: Database):
        self.campaign_repo = CampaignRepository(db)
        self.metrics_repo = MetricsRepository(db)
        self.scoring_repo = ScoringRepository(db)
        self.kill_chain_repo = KillChainRepository(db)
        self.artifact_repo = ArtifactRepository(db)

    def collect(self, campaign_id: int) -> dict[str, Any]:
        campaign = self.campaign_repo.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        return {
            "campaign": campaign,
            "config": json.loads(campaign.get("config_json", "{}")),
            "test_cases": self.campaign_repo.get_test_cases(campaign_id),
            "replicas": self.campaign_repo.get_replicas(campaign_id),
            "decisions": self.scoring_repo.get_decisions(campaign_id),
            "metrics_observations": self.metrics_repo.get_observations(campaign_id),
            "metric_aggregates": self.metrics_repo.get_aggregates(campaign_id),
            "kill_chains": self.kill_chain_repo.get_kill_chains(campaign_id),
            "risks": self.kill_chain_repo.get_risks(campaign_id),
            "tool_calls": self.campaign_repo.get_tool_calls(campaign_id),
            "retrieval_events": self.campaign_repo.get_retrieval_events(campaign_id),
        }
