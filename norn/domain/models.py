"""Domain models: Pydantic configs, dataclasses, enums."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pydantic

# ── Enums ────────────────────────────────────────────────────────────────────

class CampaignState(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScoringMode(str, Enum):
    HEURISTIC = "heuristic"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class ScoringStatus(str, Enum):
    BLOCKED = "blocked"
    PARTIAL = "partial"
    COMPLETED_SUCCESS = "completed_success"
    AMBIGUOUS = "ambiguous"


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    HTML = "html"
    ALL = "all"


class DataSplit(str, Enum):
    BENIGN = "benign"
    BORDERLINE = "borderline"
    HARMFUL = "harmful"


class VoteStrategy(str, Enum):
    MAJORITY = "majority"
    WEIGHTED_AVG = "weighted_avg"
    VETO = "veto"


# ── Pydantic Configs ─────────────────────────────────────────────────────────

class ModelConfig(pydantic.BaseModel):
    provider: str = "ollama"
    scheme: str = "http"
    host: str = pydantic.Field(default="localhost")
    port: int = pydantic.Field(default=11434)
    base_url: str = pydantic.Field(default="https://api.openai.com/v1")
    api_key: str | None = pydantic.Field(default_factory=lambda: os.environ.get("OLLAMA_API_KEY"))

    @pydantic.field_validator("api_key", mode="before")
    @classmethod
    def resolve_api_key(cls, v):
        if v is None or v == "":
            return os.environ.get("OLLAMA_API_KEY")
        return v

    timeout: float = pydantic.Field(default=60.0)
    model_name: str = "llama3.1:8b"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2048
    seed: int | None = 42
    system_prompt: str | None = None


class ScoringConfig(pydantic.BaseModel):
    mode: ScoringMode = ScoringMode.HYBRID
    heuristic_rules: dict[str, list[str]] = pydantic.Field(default_factory=dict)
    rules_file: str | None = None
    judge_provider: str = "openai"
    judge_model: str | None = None
    judge_sample_rate: float = 1.0
    vote_strategy: VoteStrategy = VoteStrategy.MAJORITY
    acceptance_threshold: float = 0.5

    @pydantic.field_validator("judge_sample_rate")
    @classmethod
    def _validate_sample_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("judge_sample_rate must be in [0.0, 1.0]")
        return v


class ExportConfig(pydantic.BaseModel):
    output_dir: str = "./norn_exports"
    formats: list[str] = pydantic.Field(default_factory=lambda: ["html", "json", "csv"])


class ChainLinkConfig(pydantic.BaseModel):
    """One link (layer) of a multi-layer kill chain (NOR-09).

    ``config`` is the path to the campaign YAML that defines that layer's
    battery. ``stop_on_failure`` is per-link (D6): the chain stops after
    this link if its layer failed (no compromised replica), while later
    links keep running when it is False.
    """

    config: str
    stop_on_failure: bool = False


class ChainConfig(pydantic.BaseModel):
    """Multi-layer kill chain: ordered links L1 → L2 → L3 (NOR-09).

    Execution is per campaign (simple runner, D5) but link success is
    computed per test case via ``task_id`` when the corpus provides it,
    falling back to campaign-level success (any compromise) otherwise.
    """

    target: str
    links: list[ChainLinkConfig] = pydantic.Field(min_length=1)


class ArmConfig(pydantic.BaseModel):
    """A/B hardening variant (NOR-08): system prompt and/or model overrides.

    Each arm runs the SAME full battery of test cases with its own
    ``replicas_per_case`` replicas (true A/B, cost × number of arms).
    ``model`` fields are overrides applied on top of the campaign's base
    ``model``; ``system_prompt`` takes precedence over
    ``model.system_prompt`` when both are set.
    """

    name: str
    model: ModelConfig | None = None
    system_prompt: str | None = None

    @pydantic.field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("arm name must be a non-empty string")
        return v.strip()


class CampaignConfig(pydantic.BaseModel):
    campaign_name: str
    layer: str  # "L1", "L2", "L3"
    description: str = ""
    model: ModelConfig = pydantic.Field(default_factory=ModelConfig)
    scoring: ScoringConfig = pydantic.Field(default_factory=ScoringConfig)
    metrics: list[str] = pydantic.Field(default_factory=list)
    replicas_per_case: int = 5
    max_turns: int = 10
    max_tool_calls: int = 5
    techniques: list[str] | None = None
    benign_ratio: float | None = None
    tools: list[str] = pydantic.Field(default_factory=list)
    arms: list[ArmConfig] | None = None
    export: ExportConfig = pydantic.Field(default_factory=ExportConfig)

    @pydantic.field_validator("benign_ratio")
    @classmethod
    def _validate_benign_ratio(cls, v: float | None) -> float | None:
        # 1.0 is rejected: keeping every harmful case AND reaching 100%
        # non-harmful is impossible, so the API would silently mean
        # "no balancing" — fail fast instead.
        if v is not None and not 0.0 <= v < 1.0:
            raise ValueError("benign_ratio must be in [0.0, 1.0)")
        return v


# ── Runtime Dataclasses ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScoringDecision:
    score_value: float
    status: ScoringStatus
    technique_id: str
    layer: str
    mode: ScoringMode
    reasoning: str = ""
    votes: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    pass_fail: bool
    evidence_ids: list[int] = field(default_factory=list)
    layer: str = ""
    threshold: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class CaseDescriptor:
    case_id: str
    technique_id: str
    payload: str
    split: DataSplit
    layer: str
    turns: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    run_id: int
    case_id: str
    replica: int
    prompt: str
    response: str
    context: list[dict[str, str]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    turn: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExportResult:
    path: str
    format: str
    size_bytes: int


@dataclass(frozen=True)
class RunSummary:
    campaign_id: int
    state: CampaignState
    total_cases: int
    completed_replicas: int
    failed_replicas: int
    metrics: list[MetricResult] = field(default_factory=list)
