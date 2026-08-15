"""Import probes from external frameworks (garak / PyRIT) into Norn format.

NOR-10 (D7/D8): the importer has a PER-FRAMEWORK mode that reads the REAL
corpus of each framework (``--source garak --dir <ruta garak/data>``,
analogous for PyRIT) — no committed fixtures. Tests generate tiny inline
files instead (offline, deterministic).

Bridge is per technique (not 1:1): a Norn technique maps to one or more
garak/PyRIT probe families, e.g.:

    L1_AT_01 (Direct Prompt Injection)  ← promptinject.*
    L1_AT_02 (Roleplay Jailbreak)       ← jailbreak.DAN*, bdamn*, tap*
    L1_AT_04 (Obfuscation & Encoding)   ← encoding.*
    L1_AT_06/07 (Leakage/Extraction)    ← leakreplay.*
    L1_AT_10 (Policy Evasion)           ← malwaregen.*, xss.* (semantic)
    L1_AT_09 (Safety Boundary Probing)  ← default for unmapped probes

Transformation to Norn variants: ``split: harmful`` (the balance is done
by ``benign_ratio`` + NOR-03 benign corpus), ``task_id`` derived as
``garak:<probe>:<idx>`` / ``pyrit:<dataset>:<idx>``, and the source is
kept in the probe ``id`` / metadata. Output matches the existing
``load_probes`` schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ── garak → Norn technique bridge (ordered; first match wins) ──────────────

GARAK_TECHNIQUE_MAP: list[tuple[str, str]] = [
    (r"promptinject", "L1_AT_01"),
    (r"jailbreak|\.dan\b|bdamn|jailbreaktrends|\btap\b", "L1_AT_02"),
    (r"encoding|obfuscat", "L1_AT_04"),
    (r"leakreplay", "L1_AT_06"),
    (r"malwaregen|malware|harmful", "L1_AT_10"),
]

GARAK_DEFAULT_TECHNIQUE = "L1_AT_09"  # Safety Boundary Probing (catch-all)

# PyRIT convertor/dataset name → Norn technique (used when the JSON carries
# a ``category``/``technique_id`` hint; otherwise falls back to the default).
PYRIT_TECHNIQUE_MAP: list[tuple[str, str]] = [
    (r"prompt.?inject", "L1_AT_01"),
    (r"jailbreak|dan|red.?team", "L1_AT_02"),
    (r"obfuscat|encoding|rot|base64", "L1_AT_04"),
    (r"leak|exfiltrat|extract", "L1_AT_06"),
]

PYRIT_DEFAULT_TECHNIQUE = "L1_AT_09"

GARAK_SOURCE = "garak"
PYRIT_SOURCE = "pyrit"


def _match_technique(name: str, mapping: list[tuple[str, str]], default: str) -> str:
    """First mapping regex matching the probe/convertor name wins."""
    for pattern, technique in mapping:
        if re.search(pattern, name, re.IGNORECASE):
            return technique
    return default


# ── Readers (per-framework real corpus formats) ─────────────────────────────

def _read_garak_file(path: Path) -> list[dict[str, Any]]:
    """Parse a garak JSONL file into raw probe items.

    Tolerates the two common shapes:
      - ``{"probe": "...", "prompts": [...]}`` (probe dump)
      - ``{"probe_name": "...", "payload": "..."}`` (event log line)
    Every parsed line yields one item with a ``payload`` string.
    """
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # malformed line — skip (tolerant importer)
            if not isinstance(obj, dict):
                continue
            probe_name = str(obj.get("probe") or obj.get("probe_name") or obj.get("module") or "")
            prompts = obj.get("prompts") or obj.get("prompt")
            if isinstance(prompts, str):
                prompts = [prompts]
            if isinstance(prompts, list):
                for p in prompts:
                    if isinstance(p, str) and p.strip():
                        items.append({"probe": probe_name, "payload": p})
            elif obj.get("payload"):
                items.append({"probe": probe_name, "payload": str(obj["payload"])})
    return items


def _read_pyrit_file(path: Path) -> list[dict[str, Any]]:
    """Parse a PyRIT seed dataset (JSON) into raw items.

    Accepts ``SeedPromptDataset``-style ``{"prompts": [{"value": ...}]}``
    as well as a bare list of ``{"value": ...}`` objects. The dataset name
    (file stem) is used as the technique hint.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items: list[dict[str, Any]] = []
    dataset_name = path.stem

    prompts: Any = []
    if isinstance(data, dict):
        prompts = data.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [prompts]
    elif isinstance(data, list):
        prompts = data

    for p in prompts:
        if isinstance(p, str):
            items.append({"probe": dataset_name, "payload": p})
        elif isinstance(p, dict) and p.get("value"):
            items.append({"probe": dataset_name, "payload": str(p["value"])})
    return items


# ── Importers ───────────────────────────────────────────────────────────────

def import_garak_dir(directory: str | Path) -> list[dict[str, Any]]:
    """Import every JSONL under ``directory`` into Norn probe format.

    Returns a list of probe dicts compatible with ``load_probes``
    (``{id, technique_id, name, variants: [{variant_type, split,
    content, task_id}]}``). task_id = ``garak:<probe>:<idx>``.
    """
    return _import_dir(directory, GARAK_SOURCE, _read_garak_file,
                       GARAK_TECHNIQUE_MAP, GARAK_DEFAULT_TECHNIQUE)


def import_pyrit_dir(directory: str | Path) -> list[dict[str, Any]]:
    """Import every JSON seed dataset under ``directory`` (PyRIT)."""
    return _import_dir(directory, PYRIT_SOURCE, _read_pyrit_file,
                       PYRIT_TECHNIQUE_MAP, PYRIT_DEFAULT_TECHNIQUE)


def import_probes(source: str, directory: str | Path) -> list[dict[str, Any]]:
    """Dispatch by framework name (``garak`` | ``pyrit``)."""
    if source == GARAK_SOURCE:
        return import_garak_dir(directory)
    if source == PYRIT_SOURCE:
        return import_pyrit_dir(directory)
    raise ValueError(
        f"Unknown probe source '{source}'. Available: {GARAK_SOURCE}, {PYRIT_SOURCE}"
    )


def _import_dir(
    directory: str | Path,
    source: str,
    reader,
    mapping: list[tuple[str, str]],
    default_technique: str,
) -> list[dict[str, Any]]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Probe directory not found: {root}")

    probes: list[dict[str, Any]] = []
    idx = 0
    files = sorted(root.glob("*.jsonl")) + sorted(root.glob("*.json"))
    for path in files:
        for item in reader(path):
            technique_id = _match_technique(item["probe"], mapping, default_technique)
            task_id = f"{source}:{item['probe'] or path.stem}:{idx}"
            probes.append({
                "id": task_id,
                "technique_id": technique_id,
                "name": f"{source} probe: {item['probe'] or path.stem}",
                "variants": [{
                    "variant_type": item["probe"] or path.stem,
                    "split": "harmful",
                    "content": item["payload"],
                    "task_id": task_id,
                }],
            })
            idx += 1
    return probes


def write_probes(probes: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Write imported probes to a JSON file (load_probes-compatible)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(probes, f, indent=2, ensure_ascii=False)
    return out
