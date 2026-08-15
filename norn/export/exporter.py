"""Multi-format export system using Strategy pattern."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from norn.domain.models import ExportResult


def _get_template_dir() -> Path:
    """Locate the Jinja2 templates directory."""
    pkg_dir = Path(__file__).resolve().parent.parent
    tmpl = pkg_dir / "reports" / "templates"
    if tmpl.exists():
        return tmpl
    fallback = Path.cwd() / "norn" / "reports" / "templates"
    return fallback


class JsonExporter:
    def export(self, data: dict[str, Any], output_dir: str, campaign_id: int) -> ExportResult:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"campaign_{campaign_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        size = os.path.getsize(path)
        return ExportResult(path=path, format="json", size_bytes=size)


class CsvExporter:
    def export(self, data: dict[str, Any], output_dir: str, campaign_id: int) -> ExportResult:
        os.makedirs(output_dir, exist_ok=True)

        test_cases = data.get("test_cases", [])
        path = os.path.join(output_dir, f"campaign_{campaign_id}_cases.csv")
        with open(path, "w", newline="") as f:
            if test_cases:
                writer = csv.DictWriter(f, fieldnames=test_cases[0].keys())
                writer.writeheader()
                writer.writerows(test_cases)
            else:
                f.write("No test cases\n")

        # NOR-07: sidecar cost file (campaign-level data doesn't fit case rows)
        cost = data.get("cost")
        if cost:
            cost_path = os.path.join(output_dir, f"campaign_{campaign_id}_cost.csv")
            with open(cost_path, "w", newline="") as f:
                fieldnames = ["model", "provider", "role", "tokens_in", "tokens_out",
                              "cost", "price_status", "currency", "source"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for line in cost.get("lines", []):
                    writer.writerow(line)
                writer.writerow({
                    "model": "TOTAL", "provider": "", "role": "",
                    "tokens_in": "", "tokens_out": "",
                    "cost": cost.get("total_cost"),
                    "price_status": "", "currency": cost.get("currency", "USD"),
                    "source": "",
                })

        # NOR-08: sidecar replicas file when the campaign used A/B arms
        # (per-arm replicas don't fit the per-case rows above).
        replicas = data.get("replicas", [])
        if replicas and any(r.get("arm") for r in replicas):
            rep_path = os.path.join(output_dir, f"campaign_{campaign_id}_replicas.csv")
            with open(rep_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=replicas[0].keys())
                writer.writeheader()
                writer.writerows(replicas)

        size = os.path.getsize(path)
        return ExportResult(path=path, format="csv", size_bytes=size)


class HtmlExporter:
    def __init__(self):
        tmpl_dir = str(_get_template_dir())
        self.env = Environment(
            loader=FileSystemLoader(tmpl_dir),
            autoescape=True,
        )

    def export(self, data: dict[str, Any], output_dir: str, campaign_id: int) -> ExportResult:
        os.makedirs(output_dir, exist_ok=True)

        template = self.env.get_template("report.html.jinja2")
        html = template.render(data=data, campaign_id=campaign_id)

        path = os.path.join(output_dir, f"campaign_{campaign_id}_report.html")
        with open(path, "w") as f:
            f.write(html)

        size = os.path.getsize(path)
        return ExportResult(path=path, format="html", size_bytes=size)


class ExportFactory:
    """Factory for resolving exporters by format string."""

    _EXPORTERS = {
        "json": JsonExporter,
        "csv": CsvExporter,
        "html": HtmlExporter,
    }

    @classmethod
    def get_exporter(cls, fmt: str):
        exporter_cls = cls._EXPORTERS.get(fmt)
        if exporter_cls is None:
            raise ValueError(f"Unknown export format: {fmt}. Available: {list(cls._EXPORTERS)}")
        return exporter_cls()

    @classmethod
    def get_all(cls) -> list:
        return [cls.get_exporter(f) for f in cls._EXPORTERS]
