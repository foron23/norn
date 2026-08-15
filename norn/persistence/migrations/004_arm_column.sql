-- NOR-08: A/B hardening — label the arm (variant) each replica belongs to.
-- NULL = no arms configured (legacy campaigns keep working unchanged).
-- The ALTER is tolerated as a no-op by migrate() when the column already
-- exists (pre-versioned databases).
ALTER TABLE run_replica ADD COLUMN arm TEXT;
