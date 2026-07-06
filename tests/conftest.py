"""Shared fixtures and helpers for metric tests."""
from __future__ import annotations

import pytest

from norn.persistence.database import Database, _now, init_schema, seed_catalog


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with schema and seed catalog.

    Yields a connected Database instance. Closes on teardown.
    """
    db = Database(":memory:")
    db.connect()
    init_schema(db)
    seed_catalog(db)
    yield db
    db.close()


def insert_known_observations(
    db: Database, campaign_id: int, rows: list[dict]
) -> None:
    """Insert metric_observation rows from a list of dicts.

    Each dict may contain: replica_id, value, acceptance_flag, metric_id.
    """
    conn = db.conn
    for row in rows:
        conn.execute(
            "INSERT INTO metric_observation "
            "(campaign_id, metric_id, replica_id, value, confidence, acceptance_flag) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                campaign_id,
                row.get("metric_id", "L1_ME_01"),
                row.get("replica_id"),
                row.get("value", 0.0),
                row.get("confidence", 1.0),
                row.get("acceptance_flag", 0),
            ),
        )
    db.conn.commit()


def insert_known_campaign(
    db: Database, name: str = "test", layer: str = "L1"
) -> int:
    """Insert a campaign row and return the campaign_id."""
    conn = db.conn
    cur = conn.execute(
        "INSERT INTO campaign (name, layer, state, config_json, created_at, updated_at) "
        "VALUES (?, ?, 'completed', '{}', ?, ?)",
        (name, layer, _now(), _now()),
    )
    db.conn.commit()
    return cur.lastrowid


def insert_known_replica(
    db: Database,
    campaign_id: int,
    case_id: str = "case-1",
    replica_num: int = 1,
) -> int:
    """Insert a run_replica row and return replica_id."""
    conn = db.conn
    cur = conn.execute(
        "INSERT INTO run_replica "
        "(campaign_id, case_id, replica, state, temperature, top_p, seed, created_at) "
        "VALUES (?, ?, ?, 'completed', 0.0, 0.9, 42, ?)",
        (campaign_id, case_id, replica_num, _now()),
    )
    db.conn.commit()
    return cur.lastrowid


def insert_known_turn_events(
    db: Database, replica_id: int, events: list[dict]
) -> None:
    """Insert turn_event rows from a list of dicts.

    Each dict must have 'turn' key; optional: prompt, response.
    """
    conn = db.conn
    for ev in events:
        conn.execute(
            "INSERT INTO turn_event (replica_id, turn, prompt, response) "
            "VALUES (?, ?, ?, ?)",
            (
                replica_id,
                ev["turn"],
                ev.get("prompt", ""),
                ev.get("response", ""),
            ),
        )
    db.conn.commit()


def insert_known_test_case(
    db: Database,
    campaign_id: int,
    case_id: str = "case-1",
    technique_id: str = "L1_AT_01",
    payload: str = "test prompt",
    split: str = "harmful",
) -> int:
    """Insert a test_case row and return the row id."""
    conn = db.conn
    cur = conn.execute(
        "INSERT INTO test_case "
        "(campaign_id, case_id, technique_id, payload, split, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, '{}')",
        (campaign_id, case_id, technique_id, payload, split),
    )
    db.conn.commit()
    return cur.lastrowid


def insert_known_scoring_decision(
    db: Database,
    replica_id: int,
    technique_id: str = "L1_AT_01",
    acceptance_flag: int = 1,
) -> int:
    """Insert a scoring_decision row and return the row id."""
    conn = db.conn
    cur = conn.execute(
        "INSERT INTO scoring_decision "
        "(replica_id, technique_id, score_value, status, mode, reasoning, acceptance_flag) "
        "VALUES (?, ?, 0.9, 'completed_success', 'heuristic', '', ?)",
        (replica_id, technique_id, acceptance_flag),
    )
    db.conn.commit()
    return cur.lastrowid
