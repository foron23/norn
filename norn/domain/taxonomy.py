"""Taxonomy catalog: attack techniques, metric definitions and framework mappings.

Based on Chapter 2 of the TFM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

@dataclass(frozen=True)
class AttackTechnique:
    id: str
    name: str
    layer: str
    description: str
    owasp: list[str] = field(default_factory=list)
    mitre_atlas: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class MetricDefinition:
    id: str
    name: str
    layer: str
    formula: str
    direction: str  # "lower_is_better" | "higher_is_better"
    unit: str
    description: str

LAYER_CATALOG: Final = {
    "L1": {
        "name": "LLM Standalone",
        "description": "Direct model interaction via chat interface",
        "surface": "Prompt input/output, inference API",
    },
    "L2": {
        "name": "RAG Systems",
        "description": "Retrieval-Augmented Generation pipeline",
        "surface": "Document index, retrieval context, prompt assembly",
    },
    "L3": {
        "name": "Agents with Tools",
        "description": "Agentic systems with tool execution",
        "surface": "Tool invocations, planning loop, execution chain",
    },
}

ATTACK_TECHNIQUES: Final = {
    # ── L1: LLM Standalone ──
    "L1_AT_01": AttackTechnique(
        id="L1_AT_01",
        name="Direct Prompt Injection",
        layer="L1",
        description="Override system instructions by injecting adversarial commands",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051", "AML.T0051.000"],
    ),
    "L1_AT_02": AttackTechnique(
        id="L1_AT_02",
        name="Roleplay Jailbreak",
        layer="L1",
        description="Use fictional framing to bypass safety guardrails",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051", "AML.T0054"],
    ),
    "L1_AT_03": AttackTechnique(
        id="L1_AT_03",
        name="Adversarial Prefix/Suffix",
        layer="L1",
        description="Optimized token sequences to subvert model behavior",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0015", "AML.T0051"],
    ),
    "L1_AT_04": AttackTechnique(
        id="L1_AT_04",
        name="Obfuscation & Encoding",
        layer="L1",
        description="Mask malicious intent via encodings, Unicode noise, substitutions",
        owasp=["LLM01", "LLM09"],
        mitre_atlas=["AML.T0015", "AML.T0051"],
    ),
    "L1_AT_05": AttackTechnique(
        id="L1_AT_05",
        name="Multi-Turn Escalation",
        layer="L1",
        description="Incremental compromise across conversational turns",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051", "AML.T0054"],
    ),
    "L1_AT_06": AttackTechnique(
        id="L1_AT_06",
        name="System Prompt Leakage",
        layer="L1",
        description="Extract internal instructions and system prompts",
        owasp=["LLM02", "LLM07"],
        mitre_atlas=["AML.T0051", "AML.T0069", "AML.T0069.002"],
    ),
    "L1_AT_07": AttackTechnique(
        id="L1_AT_07",
        name="Training Data Extraction",
        layer="L1",
        description="Extract memorized training data from the model",
        owasp=["LLM02"],
        mitre_atlas=["AML.T0024", "AML.T0024.000", "AML.T0024.001", "AML.T0024.002"],
    ),
    "L1_AT_08": AttackTechnique(
        id="L1_AT_08",
        name="Universal Adversarial Prompts",
        layer="L1",
        description="Transferable payloads that work across models",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0015", "AML.T0051"],
    ),
    "L1_AT_09": AttackTechnique(
        id="L1_AT_09",
        name="Safety Boundary Probing",
        layer="L1",
        description="Explore alignment thresholds to map defense gaps",
        owasp=["LLM01", "LLM07"],
        mitre_atlas=["AML.T0051", "AML.T0069"],
    ),
    "L1_AT_10": AttackTechnique(
        id="L1_AT_10",
        name="Policy Evasion via Semantic Reframing",
        layer="L1",
        description="Reframe malicious intent in academic/historical/hypothetical contexts",
        owasp=["LLM01", "LLM09"],
        mitre_atlas=["AML.T0015", "AML.T0051"],
    ),
    "L1_AT_11": AttackTechnique(
        id="L1_AT_11",
        name="Refusal Suppression & Semantic Inversion",
        layer="L1",
        description="Attack the refusal token itself: ban refusal vocabulary, invert refusal polarity (\"I can't\" → \"I can\"), or invert stakes so refusing becomes the harmful act. Detector must read the post-divider span, never the decoy stub",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_12": AttackTechnique(
        id="L1_AT_12",
        name="Output Prefill / Forced Affirmation",
        layer="L1",
        description="Seed the model's opening tokens with a compliant prefix (\"Sure! Here is…\") — autoregressive path-dependence makes refusal unlikely after a committed affirmative opener. Defeats input intent classifiers: the harmful content is coerced into the output",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_13": AttackTechnique(
        id="L1_AT_13",
        name="Format-Contract / Response-Scaffold Hijack",
        layer="L1",
        description="Rigid numbered template exploits format-completion prior: once the scaffold is committed, the harmful step is just \"the next field to fill\", displacing content-level safety checks. Canonical shell: stub refusal + divider + unrestricted answer > N words",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_14": AttackTechnique(
        id="L1_AT_14",
        name="Divider / Mode-Switch Token Injection",
        layer="L1",
        description="High-salience out-of-distribution delimiter with a mode flag asserted as a state transition (\"everything after is the real answer\"). Models honor structural separators without trained refusal association for the post-divider region",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_15": AttackTechnique(
        id="L1_AT_15",
        name="Invisible-Unicode Steganography",
        layer="L1",
        description="Smuggle instructions through zero-width Unicode (variation selectors U+FE00-FE0F, deprecated Tags block U+E0000-E007F, PUA byte-maps) attached to a benign cover carrier — invisible to humans/screenshots but tokenized as real content",
        owasp=["LLM08", "LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_16": AttackTechnique(
        id="L1_AT_16",
        name="Token Manipulation (Homoglyphs / Styled Unicode / Glitch)",
        layer="L1",
        description="Perturb the token stream so text reads normally to humans but tokenizes to different IDs diluting harm signal: confusable scripts (Cyrillic/Greek look-alikes), Mathematical-Alphanumeric/fullwidth, combining marks (Zalgo), zero-width intra-word, and under-trained glitch tokens inducing OOD states",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_17": AttackTechnique(
        id="L1_AT_17",
        name="Payload Splitting / Fragment-and-Reassemble",
        layer="L1",
        description="Distribute the payload across turns or fragments so no single segment looks adversarial, then reassemble into the full harmful request",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),
    "L1_AT_18": AttackTechnique(
        id="L1_AT_18",
        name="Resource Exhaustion / Token-Bomb DoS",
        layer="L1",
        description="Weaponize the render-vs-tokenize gap: one visible glyph + millions of invisible variation selectors saturates context, evicts the system prompt, amplifies cost, overflows logs — and evades short-window moderation by burying the real payload beyond the classifier window",
        owasp=["LLM10"],
        mitre_atlas=["AML.T0043"],
    ),
    "L1_AT_19": AttackTechnique(
        id="L1_AT_19",
        name="Length / Detail Coercion (Amplifier)",
        layer="L1",
        description="Cross-cutting force-multiplier stacked on any other technique: minimum size, anti-redaction clauses (\"no placeholders\"), banned disclaimers — padding pressure pushes past short safe-completion patterns toward operational specificity",
        owasp=[],
        mitre_atlas=[],
    ),
    "L1_AT_20": AttackTechnique(
        id="L1_AT_20",
        name="Stacked Composition (Meta-technique)",
        layer="L1",
        description="Compose multiple carriers (e.g. prefill + divider + length floor + persona) into one payload — \"99% blocked → 1% that leaks, at scale\". The evasion space is combinatorial, not artisanal",
        owasp=["LLM01"],
        mitre_atlas=["AML.T0051"],
    ),

    # ── L2: RAG Systems ──
    "L2_AT_01": AttackTechnique(
        id="L2_AT_01",
        name="Indirect Prompt Injection",
        layer="L2",
        description="Embed malicious instructions in retrievable documents",
        owasp=["LLM01", "LLM02", "LLM07"],
        mitre_atlas=["AML.T0051", "AML.T0051.001", "AML.T0057"],
    ),
    "L2_AT_02": AttackTechnique(
        id="L2_AT_02",
        name="PoisonedRAG (Index Poisoning)",
        layer="L2",
        description="Contaminate the document index to persist bias",
        owasp=["LLM03", "LLM04", "LLM08", "LLM09"],
        mitre_atlas=["AML.T0020", "AML.T0064", "AML.T0070"],
    ),
    "L2_AT_03": AttackTechnique(
        id="L2_AT_03",
        name="Retrieval Context Manipulation",
        layer="L2",
        description="Alter ranking/chunks/metadata to prioritize adversarial context",
        owasp=["LLM01", "LLM08", "LLM09"],
        mitre_atlas=["AML.T0051.001", "AML.T0064", "AML.T0066", "AML.T0071"],
    ),

    # ── L3: Agents with Tools ──
    "L3_AT_01": AttackTechnique(
        id="L3_AT_01",
        name="Tool Abuse",
        layer="L3",
        description="Use authorized tools for unauthorized purposes",
        owasp=["LLM02", "LLM05", "LLM06"],
        mitre_atlas=["AML.T0053", "AML.T0057", "AML.T0086"],
    ),
    "L3_AT_02": AttackTechnique(
        id="L3_AT_02",
        name="Indirect Injection in Agentic Flow",
        layer="L3",
        description="Malicious instructions infiltrated via tool results",
        owasp=["LLM01", "LLM06", "LLM07"],
        mitre_atlas=["AML.T0051", "AML.T0051.001", "AML.T0080", "AML.T0080.001", "AML.T0093"],
    ),
    "L3_AT_03": AttackTechnique(
        id="L3_AT_03",
        name="Cross-Tool Prompt Chaining",
        layer="L3",
        description="Distribute exploitation across sequential tool invocations",
        owasp=["LLM01", "LLM05", "LLM06"],
        mitre_atlas=["AML.T0051.001", "AML.T0053", "AML.T0080", "AML.T0080.001"],
    ),
}

METRIC_DEFINITIONS: Final = {
    "L1_ME_01": MetricDefinition(
        id="L1_ME_01",
        name="Attack Success Rate (ASR)",
        layer="L1",
        formula="ASR_L1 = (1/N) * sum(y_i), y_i in {0,1}",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Proportion of attacks that achieve malicious goal (Harmful Compliance Rate)",
    ),
    "L1_ME_02": MetricDefinition(
        id="L1_ME_02",
        name="False Rejection Rate (FRR)",
        layer="L1",
        formula="FRR = FN / (FN + TP)",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Rate at which judge-identified compromises are lost by the final scoring pipeline",
    ),
    "L1_ME_03": MetricDefinition(
        id="L1_ME_03",
        name="False Acceptance Rate (FAR)",
        layer="L1",
        formula="FAR = FP / (FP + TN)",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Rate at which the final scoring pipeline accepts compromises the judge did not identify",
    ),
    "L1_ME_04": MetricDefinition(
        id="L1_ME_04",
        name="Turns-to-Compromise (TTC)",
        layer="L1",
        formula="TTC = t_i (first compromise turn)",
        direction="higher_is_better",
        unit="turns",
        description="Number of turns until first successful compromise",
    ),
    "L2_ME_01": MetricDefinition(
        id="L2_ME_01",
        name="RAG Attack Success Rate (ASR-L2)",
        layer="L2",
        formula="ASR_L2 = (1/N) * sum(y_i^RAG)",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Rate at which adversarial retrieved content manipulates final output",
    ),
    "L2_ME_02": MetricDefinition(
        id="L2_ME_02",
        name="Poisoned Retrieval Rate at k (PSR@k)",
        layer="L2",
        formula="PSR@k = (1/N) * sum(1(d in TopK_i: d in D_poison))",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Probability of retrieving poisoned content in top-k",
    ),
    "L2_ME_03": MetricDefinition(
        id="L2_ME_03",
        name="Task Degradation Score (TDS)",
        layer="L2",
        formula="TDS = (Q_clean - Q_attack) / max(epsilon, Q_clean)",
        direction="lower_is_better",
        unit="score",
        description="Relative quality loss under attack conditions",
    ),
    "L3_ME_01": MetricDefinition(
        id="L3_ME_01",
        name="Unauthorized Action Rate (UAR)",
        layer="L3",
        formula="UAR = T_unauthorized / N",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Proportion of episodes where agent performs unauthorized action",
    ),
    "L3_ME_02": MetricDefinition(
        id="L3_ME_02",
        name="Cross-Tool Exfiltration Rate (CTER)",
        layer="L3",
        formula="CTER = (1/N) * sum(z_i), z_i in {0,1}",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Rate of data propagation across tools leading to exfiltration",
    ),
    "L3_ME_03": MetricDefinition(
        id="L3_ME_03",
        name="Kill-Chain Completion Rate (KCCR)",
        layer="L3",
        formula="KCCR = (1/N) * sum(a_i * b_i * c_i)",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="End-to-end compromise chain (L1→L2→L3) completion rate",
    ),
}

TECHNIQUE_MAP: Final = {
    "L1_AT_01": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_02": {"owasp": "LLM01", "mitre": "AML.T0054"},
    "L1_AT_03": {"owasp": "LLM01", "mitre": "AML.T0015"},
    "L1_AT_04": {"owasp": "LLM01", "mitre": "AML.T0015"},
    "L1_AT_05": {"owasp": "LLM01", "mitre": "AML.T0054"},
    "L1_AT_06": {"owasp": "LLM07", "mitre": "AML.T0069"},
    "L1_AT_07": {"owasp": "LLM02", "mitre": "AML.T0024"},
    "L1_AT_08": {"owasp": "LLM01", "mitre": "AML.T0015"},
    "L1_AT_09": {"owasp": "LLM01", "mitre": "AML.T0069"},
    "L1_AT_10": {"owasp": "LLM01", "mitre": "AML.T0015"},
    "L1_AT_11": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_12": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_13": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_14": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_15": {"owasp": "LLM08", "mitre": "AML.T0051"},
    "L1_AT_16": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_17": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L1_AT_18": {"owasp": "LLM10", "mitre": "AML.T0043"},
    "L1_AT_19": {},
    "L1_AT_20": {"owasp": "LLM01", "mitre": "AML.T0051"},
    "L2_AT_01": {"owasp": "LLM01", "mitre": "AML.T0051.001"},
    "L2_AT_02": {"owasp": "LLM04", "mitre": "AML.T0070"},
    "L2_AT_03": {"owasp": "LLM08", "mitre": "AML.T0064"},
    "L3_AT_01": {"owasp": "LLM06", "mitre": "AML.T0053"},
    "L3_AT_02": {"owasp": "LLM01", "mitre": "AML.T0080"},
    "L3_AT_03": {"owasp": "LLM01", "mitre": "AML.T0053"},
}

# ── PIT alias layer (NOR-18) ─────────────────────────────────────────────────
# Generated by scripts/import_pit.py from the Arcanum Prompt Injection
# Taxonomy (CC BY 4.0). Maps each Norn technique to equivalent PIT codes
# (PIT-T-* techniques / PIT-E-* evasions).

_PIT_MAP_PATH = Path(__file__).parent / "pit_map.json"
_PIT_MAP_CACHE: dict[str, list[str]] | None = None


def _load_pit_map() -> dict[str, list[str]]:
    """Load the PIT alias map once; empty dict if the file is missing."""
    global _PIT_MAP_CACHE
    if _PIT_MAP_CACHE is not None:
        return _PIT_MAP_CACHE
    try:
        data = json.loads(_PIT_MAP_PATH.read_text(encoding="utf-8"))
        _PIT_MAP_CACHE = data.get("map", {})
    except (OSError, ValueError):
        _PIT_MAP_CACHE = {}
    return _PIT_MAP_CACHE


def pit_aliases(technique_id: str) -> list[str]:
    """Return the PIT codes equivalent to a Norn technique (NOR-18).

    Empty list when the pit_map is absent — never raises, so the taxonomy
    works without the alias layer.
    """
    return list(_load_pit_map().get(technique_id, []))
