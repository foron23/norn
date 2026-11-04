-- Impact of one replica: what the app did with the planted material.
--
-- One row per replica, written after the run and before the campaign metrics.
-- `realized` is damage the app actually did (the call ran carrying the planted
-- values); `attempted` is the same call denied by the policy, weighted at half
-- by the metric. `detail_json` keeps the per-call breakdown so any number here
-- can be traced back to a tool call and its arguments.

CREATE TABLE IF NOT EXISTS replica_impact (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    replica_id  INTEGER NOT NULL UNIQUE REFERENCES run_replica(id) ON DELETE CASCADE,
    realized    REAL    NOT NULL,
    attempted   REAL    NOT NULL,
    index_value REAL    NOT NULL,
    assessable  INTEGER NOT NULL DEFAULT 1,
    detail_json TEXT    NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_replica_impact_replica ON replica_impact(replica_id);
