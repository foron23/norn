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
