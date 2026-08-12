-- NOR-07: model price catalog for per-campaign cost estimation.
-- Schema only — prices are user data managed via `norn cost set` /
-- `norn cost import --csv` (never shipped in code).
CREATE TABLE IF NOT EXISTS model_cost (
    model        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    input_per_1k REAL NOT NULL,
    output_per_1k REAL NOT NULL,
    currency     TEXT DEFAULT 'USD',
    source       TEXT,
    updated_at   TEXT,
    PRIMARY KEY (model, provider)
);
