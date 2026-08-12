-- Norn schema v1 — frozen base schema (pre-versioning state, NOR-11).
-- All statements use IF NOT EXISTS so databases created before schema
-- versioning (user_version=0) migrate without friction.

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
