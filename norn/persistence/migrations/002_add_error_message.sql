-- NOR-11 retrofit: the error_message column was previously added ad-hoc
-- inside init_schema with a try/except. Formalized here as migration 002.
-- migrate() tolerates "duplicate column name" so databases that already
-- received the column via the old inline ALTER keep working.
ALTER TABLE run_replica ADD COLUMN error_message TEXT;
