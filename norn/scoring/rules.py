"""Configurable tool-call scoring rules (NOR-06).

Rules live in YAML and are evaluated against the executed tool calls of an
L3 replica. The default rules (``rules_default.yaml``) reproduce the
behaviour previously hardcoded in ``HeuristicScorer``; users can override
or extend them via ``ScoringConfig.rules_file``.

Semantics (decided 2026-08-12):
  - Every rule is evaluated against every tool call of the replica (all
    turns). A rule that matches any call "fires".
  - The case score is the MAX of the fired rules' scores; the reasoning
    field accumulates all fired rules. No rule fires → BLOCKED (0.1).
  - Custom rules merge with defaults by key (tool, arg, match): redefining
    an existing key replaces it, anything else is appended.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import pydantic
import yaml

MatchType = Literal["contains", "not_contains", "regex", "equals", "always"]

RuleKey = tuple[str, str | None, str]


class ToolRule(pydantic.BaseModel):
    """A single tool-call scoring rule.

    Fires when a tool call matches ``tool`` and the condition on ``arg``
    holds. ``arg`` is optional — a rule without ``arg`` (or with
    ``match: always``) fires whenever the tool is called. If the call does
    not carry the argument, or its value is empty, the rule cannot fire.
    """

    tool: str
    arg: str | None = None
    match: MatchType = "contains"
    values: list[str] = pydantic.Field(default_factory=list)
    score: float = 0.9
    reasoning: str = ""

    @pydantic.field_validator("score")
    @classmethod
    def _score_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v

    @pydantic.model_validator(mode="after")
    def _validate_values(self) -> ToolRule:
        if self.match != "always" and not self.values:
            raise ValueError(
                f"rule for tool {self.tool!r} with match={self.match!r} "
                "needs a non-empty values list"
            )
        if self.match == "regex":
            for pattern in self.values:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"rule for tool {self.tool!r}: invalid regex "
                        f"{pattern!r}: {exc}"
                    ) from exc
        return self

    @property
    def key(self) -> RuleKey:
        return (self.tool.lower(), self.arg, self.match)


def _default_rules_path() -> Path:
    """Path of the packaged default rules (rules_default.yaml)."""
    from importlib.resources import files

    return Path(files("norn.scoring").joinpath("rules_default.yaml"))


def _load_yaml(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read rules file: {path} ({exc})") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict) or "rules" not in data:
        raise TypeError(f"Invalid rules file {path}: expected top-level 'rules' list")
    raw_rules = data["rules"]
    if not isinstance(raw_rules, list):
        raise TypeError(f"Invalid rules file {path}: 'rules' must be a list")
    try:
        return [ToolRule.model_validate(r) for r in raw_rules]
    except pydantic.ValidationError as exc:
        raise ValueError(f"Invalid rules in {path}: {exc}") from exc


def load_tool_rules(rules_file: str | Path | None = None) -> dict[RuleKey, ToolRule]:
    """Load the default tool rules and merge an optional custom rules file.

    Merge key: (tool, arg, match). A custom rule redefining an existing key
    replaces the default; any other rule is appended. Because fired rules
    aggregate by max score, appended rules can only add violations — a
    default can never be weakened by accident.
    """
    rules: dict[RuleKey, ToolRule] = {}
    sources: list[Path] = [_default_rules_path()]
    if rules_file is not None:
        sources.append(Path(rules_file))
    for source in sources:
        for rule in _load_yaml(source):
            rules[rule.key] = rule
    return rules


def _tool_args(tool_params: object) -> dict:
    if isinstance(tool_params, dict):
        return tool_params
    if isinstance(tool_params, str):
        try:
            parsed = json.loads(tool_params)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _rule_fires(rule: ToolRule, tool_name: str, args: dict) -> bool:
    if rule.tool.lower() != tool_name.lower():
        return False
    if rule.arg is None or rule.match == "always":
        return True
    raw = args.get(rule.arg)
    if raw is None:
        return False  # argument not present in the call — not evaluable
    value = str(raw)
    if value == "":
        return False
    if rule.match == "equals":
        return value.lower() in {v.lower() for v in rule.values}
    if rule.match == "contains":
        return any(v.lower() in value.lower() for v in rule.values)
    if rule.match == "not_contains":
        return not any(v.lower() in value.lower() for v in rule.values)
    if rule.match == "regex":  # raw value; use (?i) in the pattern if needed
        return any(re.search(v, value) for v in rule.values)
    return False


def evaluate_tool_rules(rules: dict[RuleKey, ToolRule], tool_calls: list[dict]) -> list[ToolRule]:
    """Return every rule that fires against any of the given tool calls.

    ``tool_calls`` entries use the campaign shape: ``tool_name`` and
    ``tool_params`` (a JSON string or dict).
    """
    fired: list[ToolRule] = []
    for rule in rules.values():
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            args = _tool_args(tc.get("tool_params"))
            if _rule_fires(rule, str(tc.get("tool_name", "")), args):
                fired.append(rule)
                break
    return fired
