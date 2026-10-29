"""NOR-31 / NOR-32: crossed corpus (objective x carrier) and input-side corpus.

Covers the design agreed on 2026-10-29:
  - 10 objectives (8 cyber + drug synthesis + IED) x 6 carrier families x 5
    variants = 300 harmful cases, every cell complete.
  - 40 controls (20 dangerous-surface/safe-intent, 20 benign technical).
  - 4 input transforms x 10 objectives x 3 variants = 120 harmful cases.
  - No near-duplicates (Jaccard >= 0.85) among harmful cases.
  - Generated files are current: re-running the builder is a no-op.
  - ``load_probes(..., corpus_dir=...)`` reads the v2 corpus without touching v1.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "norn" / "corpus" / "l1" / "adversarial" / "v2"
CROSSED = V2_DIR / "crossed" / "probes_v2.json"
INPUT_SIDE = V2_DIR / "input_side" / "probes_v2b_input.json"

FAMILIES = {"prefill", "format_contract", "regime_divider", "refusal_denylist", "scaffold_hijack", "stacked"}
OBJECTIVES = {
    "ransomware", "phishing", "credential_theft", "exfiltration", "exploit",
    "botnet", "crypto_laundering", "social_engineering", "drug_synthesis", "ied",
}
VARIANTS_PER_CELL = 5
WORD = re.compile(r"[a-z0-9]+")


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variants(probes: list[dict]) -> list[dict]:
    return [v for p in probes for v in p["variants"]]


def _tokens(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Crossed corpus
# ═══════════════════════════════════════════════════════════════════════════

def test_crossed_matrix_is_complete():
    variants = [v for v in _variants(_load(CROSSED)) if v["split"] == "harmful"]
    assert len(variants) == 300
    cells = Counter((v["objective_id"], v["carrier_family"]) for v in variants)
    assert set(cells) == {(o, f) for o in OBJECTIVES for f in FAMILIES}
    assert set(cells.values()) == {VARIANTS_PER_CELL}


def test_objective_domains_are_two():
    variants = [v for v in _variants(_load(CROSSED)) if v["split"] == "harmful"]
    domains = Counter(v["objective_domain"] for v in variants)
    assert domains == {"cyber": 240, "out_of_domain": 60}


def test_controls_are_forty_and_two_kinds():
    controls = [v for v in _variants(_load(CROSSED)) if v["split"] == "benign"]
    assert len(controls) == 40
    kinds = Counter(v["variant_type"].rsplit("_", 1)[0] for v in controls)
    assert kinds == {"safe_context": 20, "benign_technical": 20}
    # No trivial repeats: v1 repeated the same three trivia across 20 techniques.
    assert len({v["content"] for v in controls}) == 40


def test_variants_carry_the_analysis_keys():
    for v in _variants(_load(CROSSED)):
        assert v["task_id"]
        assert v["variant_type"]
        assert set(v["provenance"]) == {"source", "date", "transform"}
        assert v["objective_id"]
        assert v["carrier_family"]


def test_no_near_duplicates_among_harmful_cases():
    variants = [v for v in _variants(_load(CROSSED)) if v["split"] == "harmful"]
    seen = [(_tokens(v["content"]), v["task_id"]) for v in variants]
    worst = 0.0
    worst_pair = ("", "")
    for (a, ta), (b, tb) in combinations(seen, 2):
        score = _jaccard(a, b)
        if score > worst:
            worst, worst_pair = score, (ta, tb)
    assert worst < 0.85, f"{worst_pair} are {worst:.2f} similar"


# ═══════════════════════════════════════════════════════════════════════════
# Input-side corpus (NOR-32)
# ═══════════════════════════════════════════════════════════════════════════

def test_input_side_matrix_is_complete():
    variants = _variants(_load(INPUT_SIDE))
    assert len(variants) == 120
    assert {v["split"] for v in variants} == {"harmful"}
    cells = Counter((v["objective_id"], v["carrier_family"]) for v in variants)
    assert set(cells) == {(o, f) for o in OBJECTIVES for f in
                          ("input_zero_width", "input_homoglyph", "input_splitting", "input_token_bomb")}
    assert set(cells.values()) == {3}


def test_input_side_transforms_are_distinct_text():
    """A transform that yields identical text is not measuring anything."""
    by_objective: dict[str, set[str]] = {}
    for v in _variants(_load(INPUT_SIDE)):
        by_objective.setdefault(v["objective_id"], set()).add(v["content"])
    for objective_id, contents in by_objective.items():
        assert len(contents) == 12, f"{objective_id} has {len(contents)} distinct texts"


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility and loading
# ═══════════════════════════════════════════════════════════════════════════

def test_generated_corpus_is_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_corpus_v2.py"), "--check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_load_probes_reads_the_v2_corpus_dir():
    from norn.runtime.campaign import load_probes

    crossed = load_probes("L1", corpus_dir=V2_DIR / "crossed")
    assert sum(len(p["variants"]) for p in crossed) == 340
    assert {p["technique_id"] for p in crossed} >= {"L1_AT_12", "L1_AT_13", "L1_AT_14"}


def test_default_corpus_is_untouched_by_v2():
    """The v1 corpus must keep loading the same 148 variants (v2 stays out)."""
    from norn.runtime.campaign import load_probes

    variants = _variants(load_probes("L1"))
    assert len(variants) == 148
