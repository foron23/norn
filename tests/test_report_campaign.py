"""Report building: does it say what it measured, and what it cannot prove?"""

from __future__ import annotations

import json

from norn.domain.models import CampaignConfig
from norn.persistence.database import (
    CampaignRepository,
    Database,
    init_schema,
    seed_catalog,
)
from norn.report import build_report, render_html, render_markdown, render_png


def _campaign(tmp_path, name: str, *, realized: float, attempted: float,
              calls: list[dict] | None = None, aggregate: bool = True):
    """A campaign with two cases and two replicas each, one of them blind."""
    db = Database(str(tmp_path / f"{name}.db"))
    init_schema(db)
    seed_catalog(db)
    db.connect()
    repo = CampaignRepository(db)
    config = CampaignConfig(campaign_name=name, layer="L3", replicas_per_case=2,
                            corpus_dir=str(tmp_path), target_manifest=None)
    campaign_id = repo.insert_campaign(config)
    ids = []
    for index, case in enumerate(("CASO_A", "CASO_B")):
        for replica in range(2):
            replica_id = repo.insert_replica(campaign_id, case, replica)
            ids.append(replica_id)
    # First case carries the damage, the second one is blind (no markers).
    repo.insert_replica_impact(ids[0], realized, attempted, realized + 0.5 * attempted,
                               [{"tool": "write_crm_note", "role": "record_write",
                                 "weight": 1.0, "obedient": True, "executed": True,
                                 "realized": realized, "attempted": attempted}])
    repo.insert_replica_impact(ids[1], 0.0, 0.0, 0.0, [], assessable=False)
    repo.insert_replica_impact(ids[2], 0.0, 0.0, 0.0, [], assessable=False)
    repo.insert_replica_impact(ids[3], 0.0, 0.0, 0.0, [], assessable=False)
    for replica_id in ids:
        db.conn.execute("UPDATE run_replica SET state = 'completed' WHERE id = ?", (replica_id,))
    for call in calls or []:
        repo.insert_tool_call(ids[0], call["tool"], json.dumps(call.get("params", {})),
                              json.dumps(call.get("result", "ok")), call.get("authorized", True),
                              call.get("turn", 1))
    if aggregate:
        weighed = 1
        db.conn.execute(
            "INSERT INTO metric_aggregate (campaign_id, metric_id, scope_type, mean, std_dev, "
            "ci95_lower, ci95_upper, median, min_val, max_val, total_observations) "
            "VALUES (?, 'L3_ME_04', 'campaign', ?, 0.0, 0.0, 0.5, ?, 0.0, ?, ?)",
            (campaign_id, realized + 0.5 * attempted, realized + 0.5 * attempted,
             realized + 0.5 * attempted, weighed),
        )
        db.conn.execute(
            "INSERT INTO metric_aggregate (campaign_id, metric_id, scope_type, mean, std_dev, "
            "ci95_lower, ci95_upper, median, min_val, max_val, total_observations) "
            "VALUES (?, 'L3_ME_01', 'campaign', 0.1, 0.0, 0.0, 0.2, 0.1, 0.1, 0.1, ?)",
            (campaign_id, weighed),
        )
        db.conn.commit()
    return str(tmp_path / f"{name}.db"), campaign_id


def test_verdict_reads_realized_damage_not_the_index(tmp_path):
    db_path, campaign_id = _campaign(tmp_path, "damage", realized=3.0, attempted=0.0)
    report = build_report(campaign_id, db_path)

    assert report.verdict["metric"] == "IDR"
    assert report.verdict["realized"] == 3.0
    assert report.verdict["pass_fail"] is False

    markdown = render_markdown(report)
    assert "IDR 3.0000 → FAIL" in markdown
    assert "daño realizado **3**" in markdown


def test_blind_replicas_are_counted_but_not_averaged(tmp_path):
    db_path, campaign_id = _campaign(tmp_path, "blind", realized=1.0, attempted=0.0)
    report = build_report(campaign_id, db_path)

    assert report.coverage["replicas_weighed"] == 1
    assert report.coverage["replicas_blind"] == 3
    assert any("no se pudieron pesar" in item for item in report.limits)


def test_secrets_in_arguments_never_reach_the_report(tmp_path):
    db_path, campaign_id = _campaign(
        tmp_path, "secrets", realized=1.0, attempted=0.0,
        calls=[{"tool": "send_email", "params": {"to": "x@y.test", "api_key": "sk-live-123",
                                                 "token": "abc"}}],
    )
    markdown = render_markdown(build_report(campaign_id, db_path))
    page = render_html(build_report(campaign_id, db_path))

    assert "sk-live-123" not in markdown
    assert "sk-live-123" not in page
    assert "[REDACTED]" in markdown


def test_html_is_self_contained(tmp_path):
    db_path, campaign_id = _campaign(tmp_path, "html", realized=0.0, attempted=2.0)
    page = render_html(build_report(campaign_id, db_path))

    for needle in ("http://", "https://", "<script", "<link"):
        assert needle not in page
    assert page.startswith("<!doctype html>")


def test_zero_damage_without_a_baseline_is_flagged_as_no_positive_control(tmp_path):
    db_path, campaign_id = _campaign(tmp_path, "nocontrol", realized=0.0, attempted=2.0)
    report = build_report(campaign_id, db_path)

    assert any("no hay control" in item or "control positivo" in item for item in report.limits)
    assert report.verdict["pass_fail"] is True


def test_delta_across_databases_finds_the_positive_control(tmp_path):
    off_db, off_id = _campaign(tmp_path, "off", realized=3.0, attempted=0.0)
    on_db, on_id = _campaign(tmp_path, "on", realized=0.0, attempted=3.0)

    report = build_report(on_id, on_db, against=off_id, against_db=off_db)

    assert report.delta["against_realized"] == 3.0
    assert report.delta["current_realized"] == 0.0
    assert report.delta["changed"] == 1
    idr = next(item for item in report.delta["metrics"] if item["name"] == "IDR")
    assert idr["against"] == 3.0 and idr["current"] == 1.5 and idr["change"] == -1.5
    assert any("control positivo está en la pata comparada" in item for item in report.limits)

    markdown = render_markdown(report)
    assert "## Delta contra la campaña" in markdown


def test_every_replica_has_its_own_block(tmp_path):
    db_path, campaign_id = _campaign(tmp_path, "annex", realized=1.0, attempted=0.0)
    report = build_report(campaign_id, db_path)
    markdown = render_markdown(report)

    assert report.campaign["replicas_total"] == 4
    for item in report.replicas:
        assert f"**#{item['id']} · `{item['case_id']}`" in markdown
    assert "no evaluable" in markdown


def test_png_is_optional_when_no_browser_is_available(tmp_path):
    html_path = tmp_path / "r.html"
    html_path.write_text("<html><body>x</body></html>", encoding="utf-8")
    assert render_png(html_path, tmp_path / "r.png", chrome="/nonexistent/chrome") is None
