#!/usr/bin/env python3
"""Import probes from garak/PyRIT into Norn corpus format (NOR-10).

Per-framework mode (D8): reads the REAL corpus of each framework from a
directory — no committed fixtures.

Usage:
    python scripts/import_probes.py --source garak --dir /path/to/garak/data --layer L1
    python scripts/import_probes.py --source pyrit --dir /path/to/pyrit/datasets --layer L1 --out corpus/l1/adversarial/garak_import.json

The output is a JSON file compatible with ``load_probes``: each probe has
``technique_id`` (mapped via the garak/PyRIT ↔ Norn bridge), variants with
``split: harmful`` and a derived ``task_id`` (``garak:<probe>:<idx>``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from norn.corpus.importer import (
    GARAK_SOURCE,
    PYRIT_SOURCE,
    import_probes,
    write_probes,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import probes from garak/PyRIT into Norn format (NOR-10)."
    )
    parser.add_argument(
        "--source", required=True, choices=[GARAK_SOURCE, PYRIT_SOURCE],
        help="Framework whose real corpus to read.",
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing the framework corpus (JSONL for garak, JSON datasets for PyRIT).",
    )
    parser.add_argument(
        "--layer", default="L1", choices=["L1", "L2", "L3"],
        help="Target layer (techniques are mapped per layer; unmapped → L1_AT_09).",
    )
    parser.add_argument(
        "--out",
        default="corpus/l1/adversarial/garak_import.json",
        help="Output JSON path (load_probes-compatible).",
    )
    args = parser.parse_args()

    probes = import_probes(args.source, args.dir)
    if not probes:
        print(f"No probes found in {args.dir}", file=sys.stderr)
        return 1

    out = write_probes(probes, args.out)
    techniques = sorted({p["technique_id"] for p in probes})
    print(f"Imported {len(probes)} probes ({args.source}) → {out}")
    print(f"Techniques covered: {', '.join(techniques)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
