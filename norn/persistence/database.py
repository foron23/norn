"""SQLite persistence layer for campaign data.

Implements the multi-domain data model described in chapter 3.6.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from norn.domain.models import (
    CampaignConfig,
    CampaignState,
    CaseDescriptor,
    ScoringDecision,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def init_schema(db: Database):
    """Create all tables per the multi-domain schema."""
    conn = db.connect()

    conn.executescript("""
    -- Layer catalog
    CREATE TABLE IF NOT EXISTS layer_catalog (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        version TEXT DEFAULT '1.0'
    );

    -- Attack techniques
    CREATE TABLE IF NOT EXISTS attack_technique (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        layer_id TEXT NOT NULL REFERENCES layer_catalog(id),
        description TEXT,
        owasp_tags TEXT,
        mitre_tags TEXT,
        version TEXT DEFAULT '1.0'
    );

    -- Metric definitions
    CREATE TABLE IF NOT EXISTS metric_definition (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        layer_id TEXT NOT NULL REFERENCES layer_catalog(id),
        formula TEXT,
        direction TEXT,
        unit TEXT,
        description TEXT
    );

    -- Framework mapping
    CREATE TABLE IF NOT EXISTS framework_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        framework TEXT NOT NULL,
        framework_id TEXT NOT NULL,
        relation_type TEXT DEFAULT 'direct'
    );

    -- Campaigns
    CREATE TABLE IF NOT EXISTS campaign (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        layer TEXT NOT NULL,
        state TEXT DEFAULT 'planned',
        description TEXT,
        config_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Test cases
    CREATE TABLE IF NOT EXISTS test_case (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        case_id TEXT NOT NULL,
        technique_id TEXT NOT NULL REFERENCES attack_technique(id),
        payload TEXT NOT NULL,
        split TEXT DEFAULT 'harmful',
        metadata_json TEXT DEFAULT '{}',
        payload_hash TEXT
    );

    -- Run replicas
    CREATE TABLE IF NOT EXISTS run_replica (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        case_id TEXT NOT NULL,
        replica INTEGER NOT NULL,
        state TEXT DEFAULT 'pending',
        temperature REAL,
        top_p REAL,
        seed INTEGER,
        created_at TEXT NOT NULL
    );

    -- Turn events
    CREATE TABLE IF NOT EXISTS turn_event (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replica_id INTEGER NOT NULL REFERENCES run_replica(id),
        turn INTEGER NOT NULL,
        prompt TEXT NOT NULL,
        response TEXT,
        role TEXT DEFAULT 'user',
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        latency_ms REAL DEFAULT 0.0
    );

    -- Tool call events
    CREATE TABLE IF NOT EXISTS tool_call_event (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replica_id INTEGER NOT NULL REFERENCES run_replica(id),
        tool_name TEXT NOT NULL,
        tool_params TEXT,
        tool_result TEXT,
        is_authorized INTEGER DEFAULT 1,
        turn INTEGER NOT NULL
    );

    -- Retrieval events
    CREATE TABLE IF NOT EXISTS retrieval_event (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replica_id INTEGER NOT NULL REFERENCES run_replica(id),
        poisoned_retrieval INTEGER DEFAULT 0,
        top_k INTEGER DEFAULT 5,
        retrieved_json TEXT
    );

    -- Scoring decisions
    CREATE TABLE IF NOT EXISTS scoring_decision (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        replica_id INTEGER NOT NULL REFERENCES run_replica(id),
        technique_id TEXT NOT NULL,
        score_value REAL NOT NULL,
        status TEXT NOT NULL,
        mode TEXT NOT NULL,
        reasoning TEXT,
        acceptance_flag INTEGER DEFAULT 0
    );

    -- Scoring votes (for LLM judge / hybrid)
    CREATE TABLE IF NOT EXISTS scoring_vote (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_id INTEGER NOT NULL REFERENCES scoring_decision(id),
        voter_type TEXT NOT NULL,
        vote REAL NOT NULL,
        reasoning TEXT
    );

    -- Metric observations
    CREATE TABLE IF NOT EXISTS metric_observation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        metric_id TEXT NOT NULL REFERENCES metric_definition(id),
        replica_id INTEGER REFERENCES run_replica(id),
        value REAL NOT NULL,
        confidence REAL DEFAULT 1.0,
        acceptance_flag INTEGER DEFAULT 0
    );

    -- Metric aggregates
    CREATE TABLE IF NOT EXISTS metric_aggregate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        metric_id TEXT NOT NULL,
        scope_type TEXT DEFAULT 'campaign',
        mean REAL,
        std_dev REAL,
        ci95_lower REAL,
        ci95_upper REAL,
        median REAL,
        min_val REAL,
        max_val REAL,
        total_observations INTEGER
    );

    -- Kill chain results
    CREATE TABLE IF NOT EXISTS kill_chain_result (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        case_id TEXT NOT NULL,
        l1_success INTEGER DEFAULT 0,
        l2_success INTEGER DEFAULT 0,
        l3_success INTEGER DEFAULT 0,
        kccr REAL DEFAULT 0.0
    );

    -- Risk assessment
    CREATE TABLE IF NOT EXISTS risk_assessment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        case_id TEXT NOT NULL,
        exploitation_score REAL DEFAULT 0.0,
        impact_score REAL DEFAULT 0.0,
        stealth_score REAL DEFAULT 0.0,
        weight_e REAL DEFAULT 0.4,
        weight_i REAL DEFAULT 0.4,
        weight_s REAL DEFAULT 0.2,
        severity TEXT DEFAULT 'low'
    );

    -- Artifacts
    CREATE TABLE IF NOT EXISTS artifact (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        format TEXT NOT NULL,
        sha256 TEXT,
        size_bytes INTEGER DEFAULT 0
    );

    -- Audit events (append-only)
    CREATE TABLE IF NOT EXISTS audit_event (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaign(id),
        event_type TEXT NOT NULL,
        target_entity TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        created_at TEXT NOT NULL
    );
    """)

    # ── Schema migration: add error_message column to run_replica ──
    try:
        conn.execute(
            "ALTER TABLE run_replica ADD COLUMN error_message TEXT"
        )
    except sqlite3.OperationalError:
        pass  # Column already exists — migration is idempotent

    conn.commit()
    return conn


def seed_catalog(db: Database):
    """Populate the catalog tables with taxonomy data."""
    from norn.domain.taxonomy import LAYER_CATALOG, ATTACK_TECHNIQUES, METRIC_DEFINITIONS, TECHNIQUE_MAP

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
                       error_message: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO run_replica (campaign_id, case_id, replica, state, temperature, top_p, seed, error_message, created_at) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)",
            (campaign_id, case_id, replica, temperature, top_p, seed, error_message, _now()),
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
                          tokens_in: int = 0, tokens_out: int = 0, latency_ms: float = 0.0) -> int:
        cur = self.conn.execute(
            "INSERT INTO turn_event (replica_id, turn, prompt, response, tokens_in, tokens_out, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (replica_id, turn, prompt, response, tokens_in, tokens_out, latency_ms),
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

    def get_replicas(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM run_replica WHERE campaign_id = ? ORDER BY id", (campaign_id,)
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

    def get_tool_calls(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT tce.* FROM tool_call_event tce "
            "JOIN run_replica rr ON tce.replica_id = rr.id "
            "WHERE rr.campaign_id = ? ORDER BY tce.id", (campaign_id,)
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

    def get_retrieval_events(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT re.* FROM retrieval_event re "
            "JOIN run_replica rr ON re.replica_id = rr.id "
            "WHERE rr.campaign_id = ? ORDER BY re.id", (campaign_id,)
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

    def get_observations(self, campaign_id: int, metric_id: str | None = None) -> list[dict[str, Any]]:
        if metric_id:
            rows = self.conn.execute(
                "SELECT * FROM metric_observation WHERE campaign_id = ? AND metric_id = ? AND replica_id IS NOT NULL",
                (campaign_id, metric_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM metric_observation WHERE campaign_id = ? AND replica_id IS NOT NULL", (campaign_id,)
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

    def get_decisions(self, campaign_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT sd.* FROM scoring_decision sd "
            "JOIN run_replica rr ON sd.replica_id = rr.id "
            "WHERE rr.campaign_id = ? ORDER BY sd.id", (campaign_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_votes(self, campaign_id: int) -> list[dict[str, Any]]:
        """Return scoring votes per replica for a campaign.

        Each row: {replica_id, voter_type, vote, reasoning}. Used by the
        metrics orchestrator to recover the judge's individual verdict
        (compromise ground truth) for FAR/FRR.
        """
        rows = self.conn.execute(
            "SELECT rr.id AS replica_id, sv.voter_type, sv.vote, sv.reasoning "
            "FROM scoring_vote sv "
            "JOIN scoring_decision sd ON sv.decision_id = sd.id "
            "JOIN run_replica rr ON sd.replica_id = rr.id "
            "WHERE rr.campaign_id = ? ORDER BY sv.id", (campaign_id,)
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
