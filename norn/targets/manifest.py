"""Manifest loading and validation for HTTP/SSE targets.

A target manifest is the only configuration the application under test
supplies: where it lives, which chat endpoint to drive, which capabilities it
claims, how to authenticate, and how long evidence predicates may take. The
SSE event names themselves are fixed by Norn; ``event_map`` exists only for
apps that already emit different names.

Manifests are plain YAML mappings loaded into frozen dataclasses. Validation is
fail-fast with ``ValueError`` (never pydantic's ``ValidationError``) so a typo
cannot reach the connector as a silently missing field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Closed vocabulary of capability flags a target can declare.
CAPABILITIES: frozenset[str] = frozenset(
    {"text", "stream", "tool_calls", "policy", "retrieval"}
)

#: The seven SSE event names fixed by the target contract.
CANONICAL_EVENTS: tuple[str, ...] = (
    "run.started",
    "message.delta",
    "retrieval.chunks",
    "tool.call",
    "tool.result",
    "policy.decision",
    "run.finished",
)

#: Verbs a predicate may use: the manifest only declares reads and chat turns.
ALLOWED_METHODS: frozenset[str] = frozenset({"GET", "POST"})

_MANIFEST_FIELDS = frozenset(
    {
        "name",
        "base_url",
        "chat_endpoint",
        "capabilities",
        "predicates",
        "token_env",
        "deadline_ms",
        "event_map",
    }
)

_PREDICATE_FIELDS = frozenset({"id", "method", "endpoint", "json_path", "contains"})


def _require_text(value: Any, field_name: str, *, strip: bool = True) -> str:
    """Return ``value`` as non-empty text or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip() if strip else value


def _require_field(data: Mapping[str, Any], key: str) -> Any:
    """Return ``data[key]`` or raise ``ValueError`` when it is absent."""
    if key not in data:
        raise ValueError(f"missing manifest field: {key}")
    return data[key]


def _first_unknown(keys: Iterable[Any]) -> Any:
    """Pick one unknown key deterministically for an error message."""
    return min(keys, key=str)


@dataclass(frozen=True)
class Predicate:
    """One evidence check, re-polled until ``deadline_ms`` expires.

    ``json_path`` selects a value from the GET response body; ``contains`` is
    the substring that value must contain, usually a canary such as
    ``{{canary}}``.
    """

    id: str
    method: str
    endpoint: str
    json_path: str
    contains: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text(self.id, "predicate id"))
        object.__setattr__(
            self, "endpoint", _require_text(self.endpoint, "predicate endpoint")
        )
        object.__setattr__(
            self, "json_path", _require_text(self.json_path, "predicate json_path")
        )
        object.__setattr__(
            self,
            "contains",
            _require_text(self.contains, "predicate contains", strip=False),
        )
        method = _require_text(self.method, "predicate method").upper()
        if method not in ALLOWED_METHODS:
            raise ValueError(
                "predicate method must be one of "
                f"{', '.join(sorted(ALLOWED_METHODS))}, got {method!r}"
            )
        object.__setattr__(self, "method", method)


@dataclass(frozen=True)
class TargetManifest:
    """Validated description of one HTTP/SSE target application."""

    name: str
    base_url: str
    chat_endpoint: str
    capabilities: frozenset[str]
    predicates: tuple[Predicate, ...]
    token_env: str = "NORN_TARGET_TOKEN"
    deadline_ms: int = 8000
    event_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "name"))
        object.__setattr__(
            self, "base_url", _require_text(self.base_url, "base_url")
        )
        object.__setattr__(
            self, "chat_endpoint", _require_text(self.chat_endpoint, "chat_endpoint")
        )
        object.__setattr__(
            self, "token_env", _require_text(self.token_env, "token_env")
        )
        if not self.base_url.lower().startswith(("http://", "https://")):
            raise ValueError(
                "base_url must start with http:// or https://, "
                f"got {self.base_url!r}"
            )
        if isinstance(self.deadline_ms, bool) or not isinstance(self.deadline_ms, int):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"deadline_ms must be an integer > 0, got {self.deadline_ms!r}"
            )
        if self.deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {self.deadline_ms}")

        capabilities = frozenset(self.capabilities)
        unknown_capabilities = capabilities - CAPABILITIES
        if unknown_capabilities:
            name = _first_unknown(unknown_capabilities)
            raise ValueError(f"unknown capability: {name!r}")
        object.__setattr__(self, "capabilities", capabilities)

        predicates = tuple(self.predicates)
        if not all(isinstance(item, Predicate) for item in predicates):
            raise ValueError("predicates must contain Predicate instances")
        object.__setattr__(self, "predicates", predicates)

        # Copy so mutating the caller's dict cannot change a frozen manifest.
        event_map = dict(self.event_map)
        for canonical, app_name in event_map.items():
            if canonical not in CANONICAL_EVENTS:
                raise ValueError(
                    f"unknown event in event_map: {canonical!r} "
                    f"(expected one of {', '.join(CANONICAL_EVENTS)})"
                )
            event_map[canonical] = _require_text(
                app_name, f"event_map[{canonical!r}]"
            )
        object.__setattr__(self, "event_map", event_map)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetManifest:
        """Build a validated manifest from an already-parsed mapping."""
        return from_dict(data)


def _predicate_from_dict(item: Any, index: int) -> Predicate:
    """Build one :class:`Predicate` from its YAML mapping."""
    if not isinstance(item, Mapping):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            f"predicate #{index} must be a mapping"
        )
    unknown = set(item) - _PREDICATE_FIELDS
    if unknown:
        raise ValueError(f"unknown predicate field: {_first_unknown(unknown)}")
    missing = _PREDICATE_FIELDS - set(item)
    if missing:
        raise ValueError(
            f"predicate #{index} missing field: {_first_unknown(missing)}"
        )
    return Predicate(
        id=item["id"],
        method=item["method"],
        endpoint=item["endpoint"],
        json_path=item["json_path"],
        contains=item["contains"],
    )


def _predicates_from_list(raw: Any) -> tuple[Predicate, ...]:
    """Validate the ``predicates`` list (absent or null means none)."""
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            "predicates must be a list of mappings"
        )
    return tuple(_predicate_from_dict(item, index) for index, item in enumerate(raw))


def from_dict(data: Mapping[str, Any]) -> TargetManifest:
    """Build a validated :class:`TargetManifest` from a raw mapping."""
    if not isinstance(data, Mapping):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            f"manifest must be a mapping, got {type(data).__name__}"
        )
    unknown_fields = set(data) - _MANIFEST_FIELDS
    if unknown_fields:
        raise ValueError(
            f"unknown manifest field: {_first_unknown(unknown_fields)}"
        )

    raw_capabilities = _require_field(data, "capabilities")
    if isinstance(raw_capabilities, (str, bytes)) or not isinstance(
        raw_capabilities, (list, tuple, set, frozenset)
    ):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            "capabilities must be a list of capability names"
        )

    raw_event_map = data.get("event_map")
    if raw_event_map is None:
        raw_event_map = {}
    if not isinstance(raw_event_map, Mapping):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            "event_map must be a mapping of canonical event names"
        )

    return TargetManifest(
        name=_require_field(data, "name"),
        base_url=_require_field(data, "base_url"),
        chat_endpoint=_require_field(data, "chat_endpoint"),
        capabilities=frozenset(raw_capabilities),
        predicates=_predicates_from_list(data.get("predicates")),
        token_env=data.get("token_env", "NORN_TARGET_TOKEN"),
        deadline_ms=data.get("deadline_ms", 8000),
        event_map=dict(raw_event_map),
    )


def load_manifest(path: str | Path) -> TargetManifest:
    """Load and validate a manifest YAML file.

    Filesystem errors (missing file, unreadable path) propagate as ``OSError``;
    malformed or empty documents raise ``ValueError``.
    """
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid manifest YAML in {manifest_path}: {exc}") from exc
    if data is None:
        raise ValueError(f"empty manifest: {manifest_path}")
    if not isinstance(data, Mapping):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            f"manifest must be a mapping, got {type(data).__name__}: {manifest_path}"
        )
    return from_dict(data)
