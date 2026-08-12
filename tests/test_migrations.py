"""NOR-11: incremental schema migrations.

The schema version is stored in SQLite's ``PRAGMA user_version``. Migrations
live in ``norn/persistence/migrations/NNN_*.sql`` and are applied in numeric
order by ``migrate()`` — only versions above the current one, idempotently.
"""

import pytest

from norn.persistence.database import (
    Database,
    _migration_files,
    current_version,
    init_schema,
    migrate,
)

LATEST = max(version for version, _ in _migration_files())


@pytest.fixture
def fresh_db():
    db = Database(":memory:")
    db.connect()
    yield db
    db.close()


def test_migration_files_are_sorted():
    versions = [v for v, _ in _migration_files()]
    assert versions == sorted(versions)
    assert versions and versions[0] == 1


def test_fresh_db_reaches_latest_version(fresh_db: Database):
    init_schema(fresh_db)
    assert current_version(fresh_db) == LATEST
    # All base tables exist
    rows = fresh_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('campaign', 'turn_event', 'scoring_decision')"
    ).fetchall()
    assert len(rows) == 3


def test_migrate_is_idempotent(fresh_db: Database):
    init_schema(fresh_db)
    version_after_init = current_version(fresh_db)
    migrate(fresh_db)
    assert current_version(fresh_db) == version_after_init


def test_legacy_db_without_version_migrates_and_preserves_data(fresh_db: Database):
    """A pre-versioning database (user_version=0) migrates without friction."""
    conn = fresh_db.conn
    conn.execute(
        "CREATE TABLE campaign ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, layer TEXT NOT NULL, "
        "state TEXT DEFAULT 'planned', description TEXT, config_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO campaign (name, layer, config_json, created_at, updated_at) "
        "VALUES ('legacy', 'L1', '{}', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    assert current_version(fresh_db) == 0

    migrate(fresh_db)

    assert current_version(fresh_db) == LATEST
    row = conn.execute("SELECT name FROM campaign WHERE id = 1").fetchone()
    assert row["name"] == "legacy"
    # The retrofit migration added error_message to run_replica
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(run_replica)").fetchall()]
    assert "error_message" in cols


def test_legacy_db_with_error_message_column_tolerates_duplicate(fresh_db: Database):
    """A database that already ran the old inline ALTER keeps working."""
    conn = fresh_db.conn
    conn.execute(
        "CREATE TABLE run_replica ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, campaign_id INTEGER NOT NULL, "
        "case_id TEXT NOT NULL, replica INTEGER NOT NULL, state TEXT DEFAULT 'pending', "
        "temperature REAL, top_p REAL, seed INTEGER, created_at TEXT NOT NULL)"
    )
    # Old inline migration already applied the column
    conn.execute("ALTER TABLE run_replica ADD COLUMN error_message TEXT")
    conn.commit()

    migrate(fresh_db)  # must not raise on duplicate column name

    assert current_version(fresh_db) == LATEST
