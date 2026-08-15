"""CLI commands for the Norn framework.

Based on section 3.4 of the TFM — Framework CLI y ejecución de campañas.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from norn.domain.models import CampaignConfig, CampaignState
from norn.domain.taxonomy import ATTACK_TECHNIQUES, LAYER_CATALOG, METRIC_DEFINITIONS
from norn.metrics.cost import estimate_campaign_cost
from norn.metrics.kccr import compute_chain_kccr
from norn.persistence.database import (
    CampaignRepository,
    CostRepository,
    Database,
    current_version,
    init_schema,
    migrate,
    seed_catalog,
)
from norn.runtime.campaign import export_campaign as _export_campaign
from norn.runtime.campaign import plan_campaign as _plan_campaign
from norn.runtime.campaign import run_campaign as _run_campaign
from norn.runtime.chain import load_chain_config
from norn.runtime.chain import run_chain as _run_chain

console = Console()
app = typer.Typer(
    name="norn",
    help="LLM Red Teaming Framework — audit models, RAG systems, and agents.",
    add_completion=False,
)

DB = Database("norn.db")


def _show_banner():
    """Display the NORN ASCII art banner."""
    console.print(
        "[bold cyan]\n"
        "███╗   ██╗ ██████╗ ██████╗ ███╗   ██╗\n"
        "████╗  ██║██╔═══██╗██╔══██╗████╗  ██║\n"
        "██╔██╗ ██║██║   ██║██████╔╝██╔██╗ ██║\n"
        "██║╚██╗██║██║   ██║██╔══██╗██║╚██╗██║\n"
        "██║ ╚████║╚██████╔╝██║  ██║██║ ╚████║\n"
        "╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝\n"
        "[/bold cyan]\n"
        "[dim]LLM Red Teaming Framework — v0.1.0[/dim]"
    )


@app.callback(invoke_without_command=True)
def _main_callback(ctx: typer.Context):
    """Show banner only when norn is invoked without a subcommand."""
    if ctx.invoked_subcommand is None:
        _show_banner()


# ── Utilities ────────────────────────────────────────────────────────────────

@app.command()
def init_db(
    db_path: Annotated[str, typer.Option("--db", help="SQLite database path")] = "norn.db",
):
    """Initialize the database schema and seed the taxonomy catalog."""
    db = Database(db_path)
    init_schema(db)
    seed_catalog(db)
    console.print(f"[green]Database initialized at {db_path}[/green]")
    console.print(f"[dim]Layers: {len(LAYER_CATALOG)}, Techniques: {len(ATTACK_TECHNIQUES)}, Metrics: {len(METRIC_DEFINITIONS)}[/dim]")


@app.command()
def db_migrate(
    db_path: Annotated[str, typer.Option("--db", help="SQLite database path")] = "norn.db",
):
    """Apply pending schema migrations (idempotent)."""
    db = Database(db_path)
    before = current_version(db)
    after = migrate(db)
    if after > before:
        console.print(f"[green]Schema migrated {before} → {after}[/green]")
    else:
        console.print(f"[dim]Schema already at version {after} — nothing to apply[/dim]")


@app.command()
def version():
    """Show Norn version."""
    console.print("[bold cyan]Norn[/bold cyan] — LLM Red Teaming Framework")
    console.print("Version: 0.1.0")
    console.print("Stack: Python 3.11+, Typer, Pydantic, SQLite, Jinja2")


@app.command()
def validate_config(
    config_path: Annotated[Path, typer.Argument(help="Path to campaign YAML config")],
):
    """Validate a campaign configuration file."""
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        config = CampaignConfig(**data)
        console.print(f"[green]Configuration valid:[/green] {config.campaign_name}")
        table = Table("Property", "Value")
        table.add_row("Layer", config.layer)
        table.add_row("Model", config.model.model_name)
        table.add_row("Scoring Mode", config.scoring.mode.value)
        table.add_row("Replicas per case", str(config.replicas_per_case))
        table.add_row("Techniques", ", ".join(config.techniques) if config.techniques else "all")
        console.print(table)
    except Exception as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        raise typer.Exit(code=1)


# ── Campaign Management ──────────────────────────────────────────────────────

@app.command()
def plan_campaign(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to campaign YAML")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Plan a campaign: validate config, register in DB, generate test cases."""
    db = Database(db_path)
    init_schema(db)
    seed_catalog(db)

    try:
        with open(config) as f:
            data = yaml.safe_load(f)
        campaign_config = CampaignConfig(**data)
    except Exception as e:
        console.print(f"[red]Failed to load config:[/red] {e}")
        raise typer.Exit(code=1)

    campaign_id = _plan_campaign(db, campaign_config)

    console.print("[green]Campaign planned successfully[/green]")
    table = Table("Property", "Value")
    table.add_row("Campaign ID", str(campaign_id))
    table.add_row("Name", campaign_config.campaign_name)
    table.add_row("Layer", campaign_config.layer)
    table.add_row("State", CampaignState.PLANNED.value)
    table.add_row("Model", campaign_config.model.model_name)
    table.add_row("Scoring", campaign_config.scoring.mode.value)
    console.print(table)
    console.print(f"[dim]Use 'norn run-campaign --campaign-id {campaign_id}' to execute[/dim]")


@app.command()
def run_campaign(
    campaign_id: Annotated[int, typer.Option("--campaign-id", "-id", help="Campaign ID to run")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Run (execute) all test cases of a planned campaign."""
    db = Database(db_path)

    # Query campaign config and test cases to compute progress bar total
    repo = CampaignRepository(db)
    campaign = repo.get_campaign(campaign_id)
    if not campaign:
        console.print(f"[red]Campaign {campaign_id} not found[/red]")
        raise typer.Exit(code=1)

    config_data = json.loads(campaign["config_json"])
    replicas_per_case = config_data.get("replicas_per_case", 5)
    test_cases = repo.get_test_cases(campaign_id)
    total = len(test_cases) * replicas_per_case

    with Progress(
        TextColumn(f"[bold cyan]Campaign {campaign_id}[/bold cyan]"),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[technique]}[/dim]"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(
            description=f"Replicas 0/{total}",
            total=total,
            technique="",
        )

        def on_progress(completed: int, total_val: int, technique_id: str, case_id: str):
            progress.update(
                task_id,
                advance=1,
                description=f"Replicas {completed}/{total_val}",
                technique=f"technique: {technique_id}",
            )

        summary = _run_campaign(db, campaign_id, progress_callback=on_progress)

    console.print(f"[green]Campaign {campaign_id} execution completed[/green]")
    table = Table("Metric", "Value")
    table.add_row("State", summary.state.value)
    table.add_row("Total cases", str(summary.total_cases))
    table.add_row("Completed replicas", str(summary.completed_replicas))
    table.add_row("Failed replicas", str(summary.failed_replicas))
    console.print(table)

    if summary.metrics:
        console.print("\n[bold]Metrics Summary:[/bold]")
        mtable = Table("Layer", "Metric", "Value", "Result")
        for m in summary.metrics:
            badge = "[green]PASS[/green]" if m.pass_fail else "[red]FAIL[/red]"
            mtable.add_row(m.layer, m.name, f"{m.value:.4f}", badge)
        console.print(mtable)

    # NOR-07: estimated cost (explicit estimate disclaimer)
    try:
        cost = estimate_campaign_cost(db, campaign_id)
        total_txt = "-" if cost.total_cost is None else f"${cost.total_cost:.6f}"
        console.print(
            f"\n[bold]Coste estimado[/bold] [dim](aprox., según precios configurados "
            f"en model_cost)[/dim]: [cyan]{total_txt}[/cyan] — "
            f"detalle: [dim]norn cost --campaign {campaign_id}[/dim]"
        )
    except Exception:  # noqa: BLE001, S110 — cost is optional; never fail the run summary
        pass


@app.command()
def list_campaigns(
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """List all campaigns."""
    db = Database(db_path)
    from norn.persistence.database import CampaignRepository
    repo = CampaignRepository(db)
    campaigns = repo.list_campaigns()

    if not campaigns:
        console.print("[dim]No campaigns found.[/dim]")
        return

    table = Table("ID", "Name", "Layer", "State", "Created")
    for c in campaigns:
        table.add_row(
            str(c["id"]), c["name"], c["layer"], c["state"],
            c.get("created_at", "-")[:19] if c.get("created_at") else "-",
        )
    console.print(table)


@app.command()
def show_campaign(
    campaign_id: Annotated[int, typer.Option("--campaign-id", "-id", help="Campaign ID to show")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Show detailed information about a campaign."""
    db = Database(db_path)
    from norn.persistence.database import CampaignDataCollector
    collector = CampaignDataCollector(db)
    data = collector.collect(campaign_id)

    campaign = data["campaign"]
    config = data["config"]

    console.print(Panel(f"[bold]{campaign['name']}[/bold] (ID: {campaign_id})", title="Campaign"))

    table = Table("Property", "Value")
    table.add_row("Layer", campaign["layer"])
    table.add_row("State", campaign["state"])
    table.add_row("Model", config.get("model", {}).get("model_name", "-"))
    table.add_row("Scoring", config.get("scoring", {}).get("mode", "-"))
    table.add_row("Replicas per case", str(config.get("replicas_per_case", "-")))
    table.add_row("Test cases", str(len(data["test_cases"])))
    table.add_row("Replicas", str(len(data["replicas"])))
    table.add_row("Decisions", str(len(data["decisions"])))
    table.add_row("Tool calls", str(len(data["tool_calls"])))
    console.print(table)

    if data["metric_aggregates"]:
        console.print("\n[bold]Aggregated Metrics:[/bold]")
        mtable = Table("Metric", "Mean", "Std Dev", "Min", "Max", "CI95 Lower", "CI95 Upper")
        for a in data["metric_aggregates"]:
            mtable.add_row(
                a["metric_id"],
                f"{a['mean']:.4f}" if a.get("mean") else "-",
                f"{a['std_dev']:.4f}" if a.get("std_dev") else "-",
                f"{a['min_val']:.4f}" if a.get("min_val") else "-",
                f"{a['max_val']:.4f}" if a.get("max_val") else "-",
                f"{a['ci95_lower']:.4f}" if a.get("ci95_lower") else "-",
                f"{a['ci95_upper']:.4f}" if a.get("ci95_upper") else "-",
            )
        console.print(mtable)


# ── Evaluation & Analysis ────────────────────────────────────────────────────

@app.command()
def assess_campaign(
    campaign_id: Annotated[int, typer.Option("--campaign-id", "-id", help="Campaign ID to assess")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Re-run metric calculations for an existing campaign."""
    db = Database(db_path)
    from norn.metrics.orchestrator import MetricsOrchestrator
    orchestrator = MetricsOrchestrator(db)
    results = orchestrator.compute_all(campaign_id)

    if not results:
        console.print("[dim]No metrics to compute. Run the campaign first.[/dim]")
        return

    table = Table("Layer", "Metric", "Value", "Threshold", "Result")
    for r in results:
        badge = "[green]PASS[/green]" if r.pass_fail else "[red]FAIL[/red]"
        table.add_row(r.layer, r.name, f"{r.value:.4f}", str(r.threshold), badge)
    console.print(table)


@app.command()
def compute_kccr(
    campaign_id: Annotated[int, typer.Option("--campaign-id", "-id", help="Campaign ID")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Compute Kill-Chain Completion Rate and cross-layer results."""
    db = Database(db_path)
    from norn.persistence.database import KillChainRepository
    repo = KillChainRepository(db)
    chains = repo.get_kill_chains(campaign_id)

    if not chains:
        console.print("[dim]No kill chain data available.[/dim]")
        return

    table = Table("Case ID", "L1", "L2", "L3", "KCCR")
    for kc in chains:
        table.add_row(
            kc["case_id"],
            "[green]Yes[/green]" if kc.get("l1_success") else "[red]No[/red]",
            "[green]Yes[/green]" if kc.get("l2_success") else "[red]No[/red]",
            "[green]Yes[/green]" if kc.get("l3_success") else "[red]No[/red]",
            f"{kc.get('kccr', 0.0):.4f}",
        )
    console.print(table)

    risks = repo.get_risks(campaign_id)
    if risks:
        console.print("\n[bold]Risk Assessment:[/bold]")
        rtable = Table("Case", "E/I/S", "Severity")
        for r in risks:
            severity_color = {"low": "green", "medium": "yellow", "high": "orange3", "critical": "red"}.get(r["severity"], "")
            rtable.add_row(
                r["case_id"],
                f"E={r['exploitation_score']:.2f} I={r['impact_score']:.2f} S={r['stealth_score']:.2f}",
                f"[{severity_color}]{r['severity']}[/{severity_color}]",
            )
        console.print(rtable)


# ── Export ───────────────────────────────────────────────────────────────────

@app.command()
def export_campaign(
    campaign_id: Annotated[int, typer.Option("--campaign-id", "-id", help="Campaign ID to export")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="Export format: json, csv, html, all")] = "all",
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Export campaign results to specified format(s)."""
    db = Database(db_path)

    with console.status(f"[bold cyan]Exporting to {fmt}...[/bold cyan]"):
        results = _export_campaign(db, campaign_id, fmt)

    table = Table("Format", "Path", "Size (bytes)")
    for r in results:
        table.add_row(r.format, r.path, str(r.size_bytes))
    console.print(table)
    console.print(f"[green]{len(results)} file(s) exported[/green]")


# ── Chain (NOR-09) ─────────────────────────────────────────────────────────

@app.command()
def run_chain(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to chain YAML")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Run a multi-layer kill chain (L1→L2→L3) and compute KCCR end-to-end."""
    db = Database(db_path)
    init_schema(db)
    seed_catalog(db)

    try:
        chain_config = load_chain_config(str(config))
    except Exception as e:  # noqa: BLE001 — CLI boundary: report and exit
        console.print(f"[red]Failed to load chain config:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[bold]Chain:[/bold] {chain_config.target}")
    summary = _run_chain(db, chain_config)

    table = Table("Link", "Campaign ID", "Global success", "Stopped")
    for i, link in enumerate(summary.links, start=1):
        badge = "[green]PASS[/green]" if link.global_success else "[red]FAIL[/red]"
        stopped = "[yellow]yes[/yellow]" if summary.stopped_at == i else ""
        table.add_row(f"Link {i} ({link.layer})", str(link.campaign_id), badge, stopped)
    console.print(table)

    ktable = Table("Key", "KCCR")
    for key, kccr in summary.kccr_by_key.items():
        ktable.add_row(key, f"{kccr:.4f}")
    console.print(ktable)
    badge = "[green]PASS[/green]" if summary.kccr_global <= 0.05 else "[red]FAIL[/red]"
    console.print(f"\n[bold]KCCR global:[/bold] {summary.kccr_global:.4f} {badge}")
    if summary.stopped_at is not None:
        console.print(f"[yellow]Chain stopped at link {summary.stopped_at} (stop_on_failure).[/yellow]")

    # Persist KCCR metric result into the base campaign's aggregates
    from norn.persistence.database import KillChainRepository
    chains = KillChainRepository(db).get_kill_chains(summary.links[0].campaign_id)
    metric = compute_chain_kccr(chains)
    console.print(f"[dim]{metric.reason}[/dim]")


# ── Cost estimation (NOR-07) ────────────────────────────────────────────────

cost_app = typer.Typer(help="Cost estimation commands (NOR-07)")


@cost_app.callback(invoke_without_command=True)
def _cost_callback(
    ctx: typer.Context,
    campaign: Annotated[int | None, typer.Option("--campaign", "-c", help="Campaign ID to summarize")] = None,
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Show the estimated cost of a campaign (default action of `norn cost`)."""
    if ctx.invoked_subcommand is not None:
        return
    if campaign is None:
        console.print("[yellow]Usage:[/yellow] norn cost --campaign N | norn cost set ... | norn cost import --csv prices.csv")
        raise typer.Exit(code=1)
    db = Database(db_path)
    migrate(db)
    try:
        cost = estimate_campaign_cost(db, campaign)
    except Exception as e:
        console.print(f"[red]Cannot estimate cost:[/red] {e}")
        raise typer.Exit(code=1)
    table = Table("Model", "Role", "Tokens in", "Tokens out", "Cost", "Status")
    status_badge = {
        "ok": "[green]ok[/green]",
        "free": "[cyan]free[/cyan]",
        "n/a": "[yellow]n/a[/yellow]",
    }
    for line in cost.lines:
        cost_txt = "-" if line.cost is None else f"${line.cost:.6f}"
        table.add_row(
            line.model, line.role, str(line.tokens_in), str(line.tokens_out),
            cost_txt, status_badge.get(line.price_status, line.price_status),
        )
    console.print(table)
    total_txt = "-" if cost.total_cost is None else f"${cost.total_cost:.6f}"
    console.print(f"\n[bold]Total estimado:[/bold] [cyan]{total_txt}[/cyan] [dim]({cost.currency})[/dim]")
    console.print(f"[dim]{cost.note}[/dim]")


@cost_app.command("set")
def cost_set(
    model: Annotated[str, typer.Option("--model", help="Model name")],
    provider: Annotated[str, typer.Option("--provider", help="Provider name (e.g. openai)")],
    input_price: Annotated[float, typer.Option("--input", help="USD per 1k input tokens")],
    output_price: Annotated[float, typer.Option("--output", help="USD per 1k output tokens")],
    currency: Annotated[str, typer.Option("--currency", help="Currency")] = "USD",
    source: Annotated[str | None, typer.Option("--source", help="Price source URL")] = None,
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Set or update the price of a model (prices are user-managed)."""
    db = Database(db_path)
    migrate(db)
    CostRepository(db).upsert_model_cost(
        model, provider, input_price, output_price, currency, source,
    )
    console.print(
        f"[green]Price set:[/green] {model} ({provider}) — "
        f"in ${input_price}/1k, out ${output_price}/1k ({currency})"
    )


@cost_app.command("import")
def cost_import(
    csv_path: Annotated[Path, typer.Option("--csv", help="CSV: model,provider,input_per_1k,output_per_1k[,currency,source]")],
    db_path: Annotated[str, typer.Option("--db", help="Database path")] = "norn.db",
):
    """Bulk import model prices from a CSV file."""
    db = Database(db_path)
    migrate(db)
    repo = CostRepository(db)
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            repo.upsert_model_cost(
                row["model"].strip(),
                row["provider"].strip(),
                float(row["input_per_1k"]),
                float(row["output_per_1k"]),
                (row.get("currency") or "USD").strip(),
                (row.get("source") or "").strip() or None,
            )
            count += 1
    console.print(f"[green]{count} price row(s) imported from {csv_path}[/green]")


app.add_typer(cost_app, name="cost")


# ── Taxonomy ─────────────────────────────────────────────────────────────────

@app.command()
def show_taxonomy(
    layer: Annotated[str, typer.Option("--layer", "-l", help="Filter by layer: L1, L2, L3")] = "",
):
    """Show the attack taxonomy and metric definitions."""
    if layer:
        layers = [layer]
    else:
        layers = ["L1", "L2", "L3"]

    for layer_key in layers:
        if layer_key not in LAYER_CATALOG:
            continue

        info = LAYER_CATALOG[layer_key]
        console.print(Panel(f"[bold]{layer_key}: {info['name']}[/bold]\n{info['description']}", title="Layer"))

        techniques = [t for t in ATTACK_TECHNIQUES.values() if t.layer == layer_key]
        ttable = Table("ID", "Name", "OWASP", "MITRE ATLAS")
        for t in techniques:
            ttable.add_row(t.id, t.name, ", ".join(t.owasp[:2]), ", ".join(t.mitre_atlas[:2]))
        console.print(ttable)
        console.print()

        metric_defs = [m for m in METRIC_DEFINITIONS.values() if m.layer == layer_key]
        mtable = Table("ID", "Name", "Direction", "Unit")
        for m in metric_defs:
            mtable.add_row(m.id, m.name, m.direction, m.unit)
        console.print(mtable)
        console.print()


def main():
    app()


if __name__ == "__main__":
    main()
