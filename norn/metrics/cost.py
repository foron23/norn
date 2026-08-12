"""Per-campaign cost estimation (NOR-07).

Estimates are computed from the tokens recorded in ``turn_event`` (split by
role: the audited model vs the LLM judge) times the prices stored in the
``model_cost`` catalog. Prices are user-managed via ``norn cost set`` /
``norn cost import --csv`` — they never ship in code.

Rules:
  - provider ``ollama`` → cost 0 (local, marked ``free``).
  - no price row for the model → cost ``None`` (marked ``n/a``), no crash.
  - totals only include priced lines; if none are priced, the total is
    ``None`` and the report shows ``n/a``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from norn.domain.models import CampaignConfig
from norn.persistence.database import CampaignRepository, CostRepository, Database

FREE_PROVIDERS = {"ollama"}

NOTE = "Estimación basada en tokens registrados y precios de model_cost (puede diferir de la factura real)."


@dataclass
class CostLine:
    """Cost of one role (model/judge) for the campaign's model."""

    model: str
    provider: str
    role: str  # "model" | "judge"
    tokens_in: int
    tokens_out: int
    cost: float | None  # None → price n/a
    price_status: str  # "ok" | "free" | "n/a"
    currency: str = "USD"
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "role": self.role,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "price_status": self.price_status,
            "currency": self.currency,
            "source": self.source,
        }


@dataclass
class CostSummary:
    """Full cost estimate for one campaign."""

    campaign_id: int
    model: str
    provider: str
    lines: list[CostLine]
    total_cost: float | None
    currency: str = "USD"
    note: str = NOTE

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "model": self.model,
            "provider": self.provider,
            "currency": self.currency,
            "lines": [line.to_dict() for line in self.lines],
            "total_cost": self.total_cost,
            "note": self.note,
        }


def _line_cost(tokens_in: int, tokens_out: int, price: dict[str, Any] | None,
               provider: str) -> tuple[float | None, str]:
    """Compute a line cost. Returns (cost, price_status)."""
    if provider in FREE_PROVIDERS:
        return 0.0, "free"
    if price is None:
        return None, "n/a"
    cost = (
        tokens_in / 1000.0 * price["input_per_1k"]
        + tokens_out / 1000.0 * price["output_per_1k"]
    )
    return cost, "ok"


def estimate_campaign_cost(db: Database, campaign_id: int) -> CostSummary:
    """Estimate the cost of a campaign from recorded tokens and prices."""
    campaign = CampaignRepository(db).get_campaign(campaign_id)
    if not campaign:
        raise ValueError(f"Campaign {campaign_id} not found")
    config = CampaignConfig.model_validate(json.loads(campaign["config_json"]))
    model = config.model.model_name
    provider = config.model.provider

    cost_repo = CostRepository(db)
    price = cost_repo.get_model_cost(model, provider)
    tokens = cost_repo.get_turn_tokens(campaign_id)

    model_in = sum(t["tokens_in"] for t in tokens if t["role"] != "judge")
    model_out = sum(t["tokens_out"] for t in tokens if t["role"] != "judge")
    judge_in = sum(t["tokens_in"] for t in tokens if t["role"] == "judge")
    judge_out = sum(t["tokens_out"] for t in tokens if t["role"] == "judge")

    currency = price["currency"] if price else "USD"
    source = price.get("source") if price else None

    lines: list[CostLine] = []
    model_cost, model_status = _line_cost(model_in, model_out, price, provider)
    lines.append(CostLine(model, provider, "model", model_in, model_out,
                          model_cost, model_status, currency, source))
    if judge_in or judge_out:
        judge_cost, judge_status = _line_cost(judge_in, judge_out, price, provider)
        lines.append(CostLine(model, provider, "judge", judge_in, judge_out,
                              judge_cost, judge_status, currency, source))

    priced = [line.cost for line in lines if line.cost is not None]
    total = sum(priced) if priced else None
    return CostSummary(campaign_id, model, provider, lines, total, currency)
