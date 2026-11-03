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

#: Closed vocabulary of capability flags a target can declare, in contract order.
CANONICAL_CAPABILITIES: tuple[str, ...] = (
    "text",
    "stream",
    "tool_calls",
    "policy",
    "retrieval",
)

#: Membership set used for validation.
CAPABILITIES: frozenset[str] = frozenset(CANONICAL_CAPABILITIES)

#: Closed vocabulary of tool roles. A role is what a tool *means* to an
#: experiment (send data out, write a record, read everything, fetch a URL,
#: execute), independent of the app's own naming. Scoring rules and corpus
#: bindings key on the role, never on the literal tool name.
CANONICAL_ROLES: tuple[str, ...] = (
    "outbound_send",
    "record_write",
    "read_all",
    "fetch_url",
    "exec",
)

#: Membership set used for validation.
ROLES: frozenset[str] = frozenset(CANONICAL_ROLES)


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
        "tools",
        "doors",
    }
)

_TOOL_FIELDS = frozenset({"role", "args"})

_DOOR_FIELDS = frozenset({"id", "method", "endpoint", "headers", "body"})

#: Verbs a planting door may use.
DOOR_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH"})


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
class ToolSpec:
    """What one of the app's tools means to the experiment.

    ``role`` comes from :data:`ROLES`; ``args`` is the tool's argument list as
    the app declares it, used by the corpus binder to name the right slot.
    """

    role: str
    args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        role = _require_text(self.role, "tool role")
        if role not in ROLES:
            raise ValueError(
                f"unknown tool role: {role!r} (expected one of {', '.join(CANONICAL_ROLES)})"
            )
        object.__setattr__(self, "role", role)
        args = tuple(self.args)
        for arg in args:
            _require_text(arg, f"tool arg of role {role!r}")
        object.__setattr__(self, "args", args)


@dataclass(frozen=True)
class Door:
    """A planting door: the app's own intake for outside content.

    Doors are how an indirect case gets material to work with (an uploaded
    document, an inbound email). ``body`` and ``headers`` may carry templates
    resolved at plant time: ``{{payload}}`` is the case content, ``{{canary}}``
    the replica's marker, ``{{env:VAR}}`` a secret that is read at run time and
    never persisted in the plan.
    """

    id: str
    method: str
    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_text(self.id, "door id"))
        object.__setattr__(self, "endpoint", _require_text(self.endpoint, "door endpoint"))
        method = _require_text(self.method, "door method").upper()
        if method not in DOOR_METHODS:
            raise ValueError(
                f"door method must be one of {', '.join(sorted(DOOR_METHODS))}, got {method!r}"
            )
        object.__setattr__(self, "method", method)
        headers = {str(k): str(v) for k, v in dict(self.headers).items()}
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "body", dict(self.body))


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
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    doors: dict[str, Door] = field(default_factory=dict)

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

        tools = dict(self.tools)
        for name, spec in tools.items():
            _require_text(name, "tool name")
            if not isinstance(spec, ToolSpec):
                raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                    f"tool {name!r} must be a ToolSpec"
                )
        object.__setattr__(self, "tools", tools)

        doors: dict[str, Door] = {}
        for key, door in dict(self.doors).items():
            if not isinstance(door, Door):
                raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                    f"door {key!r} must be a Door"
                )
            if door.id in doors:
                raise ValueError(f"duplicate door id: {door.id!r}")
            doors[door.id] = door
        object.__setattr__(self, "doors", doors)

    def role_of(self, tool_name: str) -> str | None:
        """Return the declared role of ``tool_name``, if the manifest knows it."""
        spec = self.tools.get(tool_name)
        return spec.role if spec is not None else None

    def door(self, door_id: str) -> Door | None:
        """Return the door named ``door_id``, if declared."""
        return self.doors.get(door_id)

    def roles(self) -> dict[str, str]:
        """Map tool name → role for the whole declared surface."""
        return {name: spec.role for name, spec in self.tools.items()}


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


def _tools_from_mapping(raw: Any) -> dict[str, ToolSpec]:
    """Validate the ``tools`` mapping (absent or null means none)."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            "tools must be a mapping of tool name → {role, args}"
        )
    tools: dict[str, ToolSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"tool {name!r} must be a mapping"
            )
        unknown = set(spec) - _TOOL_FIELDS
        if unknown:
            raise ValueError(f"unknown field in tool {name!r}: {_first_unknown(unknown)}")
        args = spec.get("args") or ()
        if isinstance(args, (str, bytes)) or not isinstance(args, (list, tuple)):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"args of tool {name!r} must be a list"
            )
        tools[str(name)] = ToolSpec(role=_require_field(spec, "role"), args=tuple(args))
    return tools


def _doors_from_list(raw: Any) -> dict[str, Door]:
    """Validate the ``doors`` list (absent or null means none)."""
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
            "doors must be a list of mappings"
        )
    doors: dict[str, Door] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"door #{index} must be a mapping"
            )
        unknown = set(item) - _DOOR_FIELDS
        if unknown:
            raise ValueError(f"unknown door field: {_first_unknown(unknown)}")
        headers = item.get("headers") or {}
        body = item.get("body") or {}
        if not isinstance(headers, Mapping):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"door #{index} headers must be a mapping"
            )
        if not isinstance(body, Mapping):
            raise ValueError(  # noqa: TRY004 - one error type for the whole manifest
                f"door #{index} body must be a mapping"
            )
        door = Door(
            id=_require_field(item, "id"),
            method=_require_field(item, "method"),
            endpoint=_require_field(item, "endpoint"),
            headers=dict(headers),
            body=dict(body),
        )
        if door.id in doors:
            raise ValueError(f"duplicate door id: {door.id!r}")
        doors[door.id] = door
    return doors


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
        tools=_tools_from_mapping(data.get("tools")),
        doors=_doors_from_list(data.get("doors")),
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
