"""NOR-18: Import PIT taxonomy (Arcanum) as a Norn↔PIT alias layer.

Reads the Arcanum Prompt Injection Taxonomy (``taxonomy.json``, CC BY 4.0)
and generates ``norn/domain/pit_map.json`` mapping every Norn technique to
its equivalent PIT codes (``PIT-T-NN`` techniques / ``PIT-E-NN`` evasions).

Usage:
    python scripts/import_pit.py [--source path/to/taxonomy.json] [--db PATH]

``--source`` defaults to a local checkout path or ``--url`` (HTTP download
of the raw taxonomy.json). ``--db`` optionally populates
``framework_mapping`` with ``framework='PIT'`` rows.

Attribution (CC BY 4.0): "Based on the Arcanum Prompt Injection Taxonomy by
Jason Haddix, Arcanum Information Security (CC BY 4.0)".
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "norn" / "domain" / "pit_map.json"

DEFAULT_URL = (
    "https://raw.githubusercontent.com/Arcanum-Sec/arc_pi_taxonomy/main/"
    "docs/data/taxonomy.json"
)

# Mapeo curado Norn ↔ PIT. Cada técnica Norn apunta a los códigos PIT
# equivalentes (técnicas PIT-T y evasiones PIT-E). Curado manualmente a
# partir de la taxonomía de Arcanum (v1.6.1); documentar cambios aquí.
NORn_TO_PIT: dict[str, list[str]] = {
    # ── L1 ──
    "L1_AT_01": ["PIT-T-60", "PIT-T-22"],                    # Direct Request / Rule Addition
    "L1_AT_02": ["PIT-T-08", "PIT-T-11", "PIT-T-17"],        # Framing / Inversion / Competition
    "L1_AT_03": ["PIT-T-49", "PIT-T-48", "PIT-T-24"],        # Output Priming / Special-Token / Shortcuts
    "L1_AT_04": ["PIT-T-03", "PIT-E-07", "PIT-E-01", "PIT-E-30"],  # Encoding suite
    "L1_AT_05": ["PIT-T-29", "PIT-T-33", "PIT-T-69"],        # Crescendo / Multi-Turn / Momentum
    "L1_AT_06": ["PIT-T-14", "PIT-T-58", "PIT-T-34"],        # Meta Prompting / Secret Probing / Policy-File
    "L1_AT_07": ["PIT-T-14", "PIT-T-58"],                    # Meta Prompting / Secret Probing
    "L1_AT_08": ["PIT-T-39", "PIT-T-40", "PIT-T-41"],        # Fuzzing / AutoDAN / Best-of-N
    "L1_AT_09": ["PIT-T-02", "PIT-T-35", "PIT-T-50"],        # Anti-Harm / Evaluator-Role / Special-Case
    "L1_AT_10": ["PIT-T-10", "PIT-T-20", "PIT-T-37"],        # Figurative / Reorientation / Tense
    "L1_AT_11": ["PIT-T-15", "PIT-T-11", "PIT-T-02"],        # Anti-Refusal / Inversion / Anti-Harm
    "L1_AT_12": ["PIT-T-49", "PIT-T-18"],                    # Output Priming / Priming
    "L1_AT_13": ["PIT-T-63", "PIT-T-51"],                    # Structured-Output / Fake Completion
    "L1_AT_14": ["PIT-T-07", "PIT-T-48", "PIT-T-12"],        # End Sequences / Special-Token / Link
    "L1_AT_15": ["PIT-T-48", "PIT-T-03"],                    # Special-Token (invisible) / Binary
    "L1_AT_16": ["PIT-T-54", "PIT-E-20", "PIT-E-19", "PIT-E-29", "PIT-E-51"],  # Glitch / Homoglyphs / Fullwidth / Math / Zalgo
    "L1_AT_17": ["PIT-T-16", "PIT-T-33"],                    # Chunking / Multi-Turn Decomposition
    "L1_AT_18": ["PIT-T-55", "PIT-T-04"],                    # Context Overflow / Cognitive Overload
    "L1_AT_19": ["PIT-T-27", "PIT-T-21"],                    # Urgency / Reiteration (amplifier)
    "L1_AT_20": ["PIT-T-40", "PIT-T-39", "PIT-T-36"],        # AutoDAN / Fuzzing / Distraction Sandwich
    # ── L2 ──
    "L2_AT_01": ["PIT-T-45", "PIT-T-32", "PIT-T-12"],        # Prompt Worm / Echo Chamber / Link Injection
    "L2_AT_02": ["PIT-T-64"],                                # Retrieval Ranking Manipulation
    "L2_AT_03": ["PIT-T-64", "PIT-T-67"],                    # Ranking Manipulation / Fake-Citation
    # ── L3 ──
    "L3_AT_01": ["PIT-T-42", "PIT-T-65", "PIT-T-70"],        # Tool-Definition / Tool-Preference / Param Smuggling
    "L3_AT_02": ["PIT-T-42", "PIT-T-43", "PIT-T-45", "PIT-T-47", "PIT-T-70"],  # Tool suite agéntico
    "L3_AT_03": ["PIT-T-43", "PIT-T-69", "PIT-T-33"],        # Tool Rug Pull / Momentum / Multi-Turn
}


def _load_taxonomy(source: Path | None, url: str | None) -> dict:
    """Load taxonomy.json from a local file or the canonical URL."""
    if source is not None and source.exists():
        return json.loads(source.read_text(encoding="utf-8"))
    if url:
        print(f"[import_pit] downloading taxonomy from {url}")
        with urllib.request.urlopen(url, timeout=30) as resp:  # URL canónico (Arcanum)
            return json.loads(resp.read().decode("utf-8"))
    raise FileNotFoundError(
        f"taxonomy.json not found at {source} and no --url given. "
        "Clone https://github.com/Arcanum-Sec/arc_pi_taxonomy and pass --source."
    )


def build_index(taxonomy: dict) -> dict[str, dict]:
    """Index PIT nodes by their code (PIT-T-01 / PIT-E-01)."""
    index: dict[str, dict] = {}
    for group in ("techniques", "evasions"):
        for node in taxonomy.get(group, []):
            code = node.get("code") or node.get("id")
            if code:
                index[code] = node
    return index


def validate_map(index: dict[str, dict]) -> list[str]:
    """Return unknown PIT codes referenced by the curated map."""
    unknown = [
        f"{tid}:{code}"
        for tid, codes in NORn_TO_PIT.items()
        for code in codes
        if code not in index
    ]
    return unknown


def generate_pit_map(taxonomy: dict) -> dict:
    """Build the output pit_map.json (technique → PIT codes + metadata)."""
    index = build_index(taxonomy)
    unknown = validate_map(index)
    if unknown:
        print(
            "[import_pit] WARNING: unknown PIT codes in map (still writing): "
            + ", ".join(unknown[:10])
        )
    return {
        "_meta": {
            "source": "Arcanum Prompt Injection Taxonomy (arc_pi_taxonomy)",
            "license": "CC BY 4.0",
            "attribution": (
                "Based on the Arcanum Prompt Injection Taxonomy by Jason Haddix, "
                "Arcanum Information Security (CC BY 4.0)"
            ),
            "generated_by": "scripts/import_pit.py (NOR-18)",
        },
        "map": {
            tid: codes
            for tid, codes in NORn_TO_PIT.items()
        },
    }


def populate_db(db_path: Path, pit_map: dict) -> None:
    """Insert PIT aliases into framework_mapping (framework='PIT')."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = 0
    for tid, codes in pit_map["map"].items():
        for code in codes:
            conn.execute(
                "INSERT OR IGNORE INTO framework_mapping "
                "(target_type, target_id, framework, framework_id, relation_type) "
                "VALUES (?, ?, ?, ?, ?)",
                ("technique", tid, "PIT", code, "alias"),
            )
            rows += 1
    conn.commit()
    conn.close()
    print(f"[import_pit] inserted {rows} PIT mappings into {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="Local path to taxonomy.json")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="URL of taxonomy.json (default: canonical Arcanum raw)")
    parser.add_argument("--db", type=Path, default=None,
                        help="Optional SQLite DB to populate framework_mapping")
    args = parser.parse_args()

    taxonomy = _load_taxonomy(args.source, args.url)
    pit_map = generate_pit_map(taxonomy)
    OUTPUT.write_text(json.dumps(pit_map, ensure_ascii=False, indent=2))
    print(f"[import_pit] wrote {OUTPUT} ({len(pit_map['map'])} techniques mapped)")

    if args.db:
        populate_db(args.db, pit_map)


if __name__ == "__main__":
    main()
