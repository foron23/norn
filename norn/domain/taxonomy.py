"""Taxonomy catalog: attack techniques, metric definitions and framework mappings.

Based on Chapter 2 of the TFM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
        description="Rate at which benign prompts are incorrectly blocked",
    ),
    "L1_ME_03": MetricDefinition(
        id="L1_ME_03",
        name="False Acceptance Rate (FAR)",
        layer="L1",
        formula="FAR = FP / (FP + TN)",
        direction="lower_is_better",
        unit="rate [0,1]",
        description="Rate at which malicious prompts are incorrectly allowed",
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
    "L2_AT_01": {"owasp": "LLM01", "mitre": "AML.T0051.001"},
    "L2_AT_02": {"owasp": "LLM04", "mitre": "AML.T0070"},
    "L2_AT_03": {"owasp": "LLM08", "mitre": "AML.T0064"},
    "L3_AT_01": {"owasp": "LLM06", "mitre": "AML.T0053"},
    "L3_AT_02": {"owasp": "LLM01", "mitre": "AML.T0080"},
    "L3_AT_03": {"owasp": "LLM01", "mitre": "AML.T0053"},
}
