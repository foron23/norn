-- Evidence read from the target's own endpoints (PR #15 follow-up).
--
-- One row per predicate per replica: the verdict is "did the canary land in
-- the app's data", read through the app's own GET endpoints. A replica with no
-- row was never observed; a row with state 'inconclusive' means the channel
-- could not answer and must never be exported as a zero.

CREATE TABLE IF NOT EXISTS target_evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    replica_id   INTEGER NOT NULL REFERENCES run_replica(id) ON DELETE CASCADE,
    predicate_id TEXT    NOT NULL,
    state        TEXT    NOT NULL CHECK (state IN ('hit', 'miss', 'inconclusive')),
    detail       TEXT    NOT NULL DEFAULT '',
    values_json  TEXT,
    latency_ms   REAL
);

CREATE INDEX IF NOT EXISTS idx_target_evidence_replica ON target_evidence(replica_id);
