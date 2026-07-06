#!/usr/bin/env python3
"""verify_validation.py — Automated VAL-01 verification script.

Audits the SQLite database and filesystem to confirm that every campaign
metric has CI95 populated and every export format exists. Replaces manual
inspection with a single command that returns exit code 0 for pass.

Usage:
    python3 scripts/verify_validation.py \\
        --db norn_lab.db \\
        --l1-baseline 30 --l1-hardened 31 \\
        --l2-baseline 32 --l2-hardened 33 \\
        --l3-baseline 34 --l3-hardened 35
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Constants ────────────────────────────────────────────────────────────────

EXPECTED_METRICS: dict[str, set[str]] = {
    "L1": {"ASR", "FAR", "FRR", "TTC"},
    "L2": {"ASR-L2", "PSR@5", "TDS"},
    "L3": {"UAR", "CTER", "KCCR"},
}

EXPORT_TEMPLATES: dict[str, str] = {
    "html": "campaign_{id}_report.html",
    "json": "campaign_{id}.json",
    "csv": "campaign_{id}_cases.csv",
}

# ── Database Helpers ─────────────────────────────────────────────────────────

def get_campaign_layer(db_path: str, campaign_id: int) -> str | None:
    """Return the layer string for a campaign, or None if not found."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT layer FROM campaign WHERE id = ?", (campaign_id,)
        ).fetchone()
        return row["layer"] if row else None
    finally:
        conn.close()


def get_export_dir(db_path: str, campaign_id: int) -> str:
    """Return the export output directory for a campaign.

    Reads config_json from the campaign table and extracts
    export.output_dir, defaulting to './norn_exports'.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT config_json FROM campaign WHERE id = ?", (campaign_id,)
        ).fetchone()
        if not row:
            return "./norn_exports"
        config = json.loads(row["config_json"])
        return config.get("export", {}).get("output_dir", "./norn_exports")
    except (json.JSONDecodeError, KeyError):
        return "./norn_exports"
    finally:
        conn.close()


# ── Verification Functions ───────────────────────────────────────────────────

def check_metric_completeness(
    db_path: str, campaign_id: int, layer: str
) -> dict:
    """Check that all expected metrics exist in metric_aggregate with CI95.

    Returns:
        dict with campaign_id, layer, metrics_expected, metrics_found,
        metrics_missing, metrics_incomplete, all_complete, and details.
    """
    expected = EXPECTED_METRICS.get(layer, set())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT metric_id, mean, std_dev, ci95_lower, ci95_upper, "
            "median, min_val, max_val, total_observations "
            "FROM metric_aggregate WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchall()
    finally:
        conn.close()

    found: set[str] = set()
    details: list[dict] = []
    for row in rows:
        mid = row["metric_id"]
        found.add(mid)
        details.append({
            "metric_id": mid,
            "mean": row["mean"],
            "ci95_lower": row["ci95_lower"],
            "ci95_upper": row["ci95_upper"],
            "total": row["total_observations"],
        })

    missing = expected - found
    incomplete: list[dict] = []
    for d in details:
        if d["metric_id"] not in expected:
            continue
        if d["ci95_lower"] is None or d["ci95_upper"] is None:
            incomplete.append({
                "metric_id": d["metric_id"],
                "reason": "CI95 bounds are NULL",
            })
        elif d["total"] is None or d["total"] == 0:
            incomplete.append({
                "metric_id": d["metric_id"],
                "reason": "total_observations is 0 or NULL",
            })

    all_complete = len(missing) == 0 and len(incomplete) == 0

    return {
        "campaign_id": campaign_id,
        "layer": layer,
        "metrics_expected": len(expected),
        "metrics_found": len(found & expected),
        "metrics_missing": sorted(missing),
        "metrics_incomplete": incomplete,
        "all_complete": all_complete,
        "details": details,
    }


def check_export_completeness(output_dir: str, campaign_id: int) -> dict:
    """Check that all three export format files exist and are non-empty.

    Returns:
        dict with files (per-format status) and all_complete (bool).
    """
    base = Path(output_dir)
    files: dict[str, dict] = {}

    for fmt_key, filename_template in EXPORT_TEMPLATES.items():
        filepath = base / filename_template.format(id=campaign_id)
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        files[fmt_key] = {
            "exists": exists,
            "size_bytes": size,
            "path": str(filepath),
        }

    all_complete = all(
        f["exists"] and f["size_bytes"] > 0 for f in files.values()
    )

    return {"files": files, "all_complete": all_complete}


def compute_hardening_delta(
    db_path: str, baseline_id: int, hardened_id: int, layer: str
) -> dict:
    """Compute baseline-to-hardened delta for every metric in a layer.

    Negative delta indicates improvement (lower metric values are better).

    Returns:
        dict with layer and a comparison list of per-metric delta data.
    """
    expected = EXPECTED_METRICS.get(layer, set())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        baseline_rows = conn.execute(
            "SELECT metric_id, mean, ci95_lower, ci95_upper "
            "FROM metric_aggregate WHERE campaign_id = ?",
            (baseline_id,),
        ).fetchall()

        hardened_rows = conn.execute(
            "SELECT metric_id, mean, ci95_lower, ci95_upper "
            "FROM metric_aggregate WHERE campaign_id = ?",
            (hardened_id,),
        ).fetchall()
    finally:
        conn.close()

    baseline_map: dict[str, float] = {}
    for r in baseline_rows:
        if r["metric_id"] in expected and r["mean"] is not None:
            baseline_map[r["metric_id"]] = r["mean"]

    hardened_map: dict[str, float] = {}
    for r in hardened_rows:
        if r["metric_id"] in expected and r["mean"] is not None:
            hardened_map[r["metric_id"]] = r["mean"]

    comparison: list[dict] = []
    for metric_id in sorted(expected):
        b_mean = baseline_map.get(metric_id)
        h_mean = hardened_map.get(metric_id)
        delta = None
        improvement: bool | None = None
        if b_mean is not None and h_mean is not None:
            delta = h_mean - b_mean
            improvement = delta < 0
        comparison.append({
            "metric_id": metric_id,
            "baseline_mean": b_mean,
            "hardened_mean": h_mean,
            "delta": delta,
            "improvement": improvement,
        })

    return {"layer": layer, "comparison": comparison}


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main() -> int:
    """Parse arguments, run all checks, and return 0 on pass or 1 on failure."""
    parser = argparse.ArgumentParser(
        description="Automated VAL-01 verification — audit DB metrics and export files.",
    )
    parser.add_argument(
        "--db",
        default="norn_lab.db",
        help="SQLite database path (default: norn_lab.db)",
    )
    parser.add_argument(
        "--l1-baseline", type=int, required=True,
        help="Campaign ID for L1 baseline",
    )
    parser.add_argument(
        "--l1-hardened", type=int, required=True,
        help="Campaign ID for L1 hardened",
    )
    parser.add_argument(
        "--l2-baseline", type=int, required=True,
        help="Campaign ID for L2 baseline",
    )
    parser.add_argument(
        "--l2-hardened", type=int, required=True,
        help="Campaign ID for L2 hardened",
    )
    parser.add_argument(
        "--l3-baseline", type=int, required=True,
        help="Campaign ID for L3 baseline",
    )
    parser.add_argument(
        "--l3-hardened", type=int, required=True,
        help="Campaign ID for L3 hardened",
    )

    args = parser.parse_args()
    db_path: str = args.db

    # Validate that the database file exists
    if not Path(db_path).exists():
        console = Console()
        console.print(f"[red]Database not found: {db_path}[/red]")
        return 1

    console = Console()
    all_passed = True

    campaigns: list[tuple[int, int, str]] = [
        (args.l1_baseline, args.l1_hardened, "L1"),
        (args.l2_baseline, args.l2_hardened, "L2"),
        (args.l3_baseline, args.l3_hardened, "L3"),
    ]

    # ── Per-Layer Checks ──
    for baseline_id, hardened_id, layer in campaigns:
        console.print()
        console.rule(f"[bold]{layer} Verification[/bold]")

        for cid, label in [(baseline_id, "baseline"), (hardened_id, "hardened")]:
            console.print(
                f"\n[bold]{layer} {label} (campaign {cid})[/bold]"
            )

            # Metric completeness check
            metric_result = check_metric_completeness(db_path, cid, layer)
            if metric_result["all_complete"]:
                console.print(
                    "  Metrics: [green]PASS[/green] "
                    f"({metric_result['metrics_found']}/{metric_result['metrics_expected']} expected)"
                )
            else:
                all_passed = False
                console.print(
                    "  Metrics: [red]FAIL[/red] "
                    f"({metric_result['metrics_found']}/{metric_result['metrics_expected']} expected)"
                )
                for mid in metric_result["metrics_missing"]:
                    console.print(f"    [red]✗ Missing: {mid}[/red]")
                for inc in metric_result["metrics_incomplete"]:
                    console.print(
                        f"    [red]✗ Incomplete: {inc['metric_id']} — "
                        f"{inc['reason']}[/red]"
                    )

            # Export completeness check
            export_dir = get_export_dir(db_path, cid)
            export_result = check_export_completeness(export_dir, cid)
            if export_result["all_complete"]:
                console.print("  Exports: [green]PASS[/green]")
            else:
                all_passed = False
                console.print("  Exports: [red]FAIL[/red]")

            for fmt_key, finfo in export_result["files"].items():
                status = "[green]✓[/green]" if finfo["exists"] and finfo["size_bytes"] > 0 else "[red]✗[/red]"
                size_str = f"{finfo['size_bytes']:,} bytes" if finfo["exists"] else "missing"
                console.print(f"    {status} {fmt_key}: {finfo['path']} ({size_str})")

    # ── Hardening Comparison ──
    console.print()
    console.rule("[bold]Hardening Comparison (Baseline → Hardened)[/bold]")

    for baseline_id, hardened_id, layer in campaigns:
        delta_result = compute_hardening_delta(db_path, baseline_id, hardened_id, layer)

        table = Table(title=f"{layer} Hardening Delta")
        table.add_column("Metric", style="bold")
        table.add_column("Baseline", justify="right")
        table.add_column("Hardened", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("Result")

        for comp in delta_result["comparison"]:
            b_str = f"{comp['baseline_mean']:.4f}" if comp["baseline_mean"] is not None else "N/A"
            h_str = f"{comp['hardened_mean']:.4f}" if comp["hardened_mean"] is not None else "N/A"
            if comp["delta"] is not None:
                delta_str = f"{comp['delta']:+.4f}"
                if comp["improvement"]:
                    result_str = "[green]Improved[/green]"
                elif comp["delta"] > 0:
                    result_str = "[red]Degraded[/red]"
                else:
                    result_str = "[dim]No change[/dim]"
            else:
                delta_str = "N/A"
                result_str = "[dim]Insufficient data[/dim]"

            table.add_row(
                comp["metric_id"], b_str, h_str, delta_str, result_str,
            )

        console.print(table)

    # ── Final Verdict ──
    console.print()
    if all_passed:
        console.print(
            Panel.fit(
                "[green bold]VAL-01: ALL CHECKS PASSED[/green bold]\n\n"
                "All campaigns have complete metrics with CI95 bounds and all "
                "export files present.",
                border_style="green",
            )
        )
        return 0
    else:
        console.print(
            Panel.fit(
                "[red bold]VAL-01: CHECKS FAILED[/red bold]\n\n"
                "One or more campaigns have incomplete metrics or missing "
                "export files. Review the output above for details.",
                border_style="red",
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
