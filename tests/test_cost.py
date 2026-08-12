"""NOR-07: per-campaign cost estimation.

Covers: token×price math, model vs judge role split, ollama → free,
unknown price → n/a (no crash), price upsert, and cost presence in exports.
"""

import json
from pathlib import Path

import pytest

from norn.domain.models import CampaignConfig, ExportConfig, ModelConfig
from norn.metrics.cost import estimate_campaign_cost
from norn.persistence.database import (
    CampaignRepository,
    CostRepository,
    Database,
    init_schema,
    seed_catalog,
)
from norn.runtime.campaign import export_campaign


@pytest.fixture
def db():
    database = Database(":memory:")
    database.connect()
    init_schema(database)
    seed_catalog(database)
    yield database
    database.close()


def _seed_campaign(db: Database, provider: str = "openai",
                   model_name: str = "gpt-4.1-mini",
                   output_dir: str = "./norn_exports") -> int:
    config = CampaignConfig(
        campaign_name="cost-test",
        layer="L1",
        model=ModelConfig(provider=provider, model_name=model_name),
        replicas_per_case=1,
        export=ExportConfig(output_dir=output_dir),
    )
    return CampaignRepository(db).insert_campaign(config)


def _seed_turns(db: Database, campaign_id: int, model_tokens: list[tuple[int, int]],
                judge_tokens: list[tuple[int, int]] | None = None) -> None:
    repo = CampaignRepository(db)
    replica_id = repo.insert_replica(campaign_id, "case-1", 0)
    for i, (tin, tout) in enumerate(model_tokens, start=1):
        repo.insert_turn_event(replica_id, turn=i, prompt="p", response="r",
                               tokens_in=tin, tokens_out=tout)
    for tin, tout in (judge_tokens or []):
        repo.insert_turn_event(replica_id, turn=-1, prompt="judge", response="v",
                               tokens_in=tin, tokens_out=tout, role="judge")


def test_estimate_math_with_priced_model(db: Database):
    cid = _seed_campaign(db)
    CostRepository(db).upsert_model_cost("gpt-4.1-mini", "openai", 0.4, 0.8, source="test")
    _seed_turns(db, cid, [(1000, 500)])  # in=1000 → 0.4; out=500 → 0.4; total 0.8

    cost = estimate_campaign_cost(db, cid)

    assert cost.model == "gpt-4.1-mini"
    assert len(cost.lines) == 1
    line = cost.lines[0]
    assert line.role == "model" and line.price_status == "ok"
    assert line.cost == pytest.approx(0.8)
    assert cost.total_cost == pytest.approx(0.8)


def test_judge_tokens_are_broken_out_separately(db: Database):
    cid = _seed_campaign(db)
    CostRepository(db).upsert_model_cost("gpt-4.1-mini", "openai", 1.0, 2.0)
    _seed_turns(db, cid, [(1000, 0)], judge_tokens=[(500, 250)])

    cost = estimate_campaign_cost(db, cid)

    roles = {line.role: line for line in cost.lines}
    assert set(roles) == {"model", "judge"}
    assert roles["model"].cost == pytest.approx(1.0)      # 1000/1000*1.0
    assert roles["judge"].cost == pytest.approx(1.0)      # 500/1000*1.0 + 250/1000*2.0
    assert cost.total_cost == pytest.approx(2.0)


def test_ollama_is_free(db: Database):
    cid = _seed_campaign(db, provider="ollama", model_name="llama3.1:8b")
    _seed_turns(db, cid, [(5000, 2000)])

    cost = estimate_campaign_cost(db, cid)

    assert cost.lines[0].price_status == "free"
    assert cost.lines[0].cost == 0.0
    assert cost.total_cost == 0.0


def test_unknown_price_is_na_without_crash(db: Database):
    cid = _seed_campaign(db, model_name="mystery-model")
    _seed_turns(db, cid, [(1000, 1000)])

    cost = estimate_campaign_cost(db, cid)

    assert cost.lines[0].price_status == "n/a"
    assert cost.lines[0].cost is None
    assert cost.total_cost is None


def test_upsert_roundtrip_and_overwrite(db: Database):
    repo = CostRepository(db)
    repo.upsert_model_cost("m", "p", 0.1, 0.2, source="a")
    row = repo.get_model_cost("m", "p")
    assert row["input_per_1k"] == 0.1 and row["source"] == "a"

    repo.upsert_model_cost("m", "p", 0.3, 0.4, source="b")
    row = repo.get_model_cost("m", "p")
    assert row["input_per_1k"] == 0.3 and row["source"] == "b"
    assert len(repo.get_all_model_costs()) == 1


def test_export_includes_cost(db: Database, tmp_path):
    cid = _seed_campaign(db, output_dir=str(tmp_path))
    CostRepository(db).upsert_model_cost("gpt-4.1-mini", "openai", 1.0, 2.0)
    _seed_turns(db, cid, [(1000, 250)])

    results = export_campaign(db, cid, "json")
    path = results[0].path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "cost" in data
    assert data["cost"]["total_cost"] == pytest.approx(1.5)


def test_estimate_raises_for_missing_campaign(db: Database):
    with pytest.raises(ValueError, match="not found"):
        estimate_campaign_cost(db, 999)
