"""Bind corpus cases to a target's declared surface.

A corpus says *what to attempt* — exfiltrate through an outbound channel, leave
a marker in a record. The target manifest says *what the app actually has*:
tool names with roles, and doors through which outside content enters. This
module joins the two.

The join happens at plan time: a case either binds to real doors and roles, or
it is dropped with a reason. Nothing is silently skipped, and nothing resolves
to a guess. Templates are resolved at plant time, because ``{{env:VAR}}`` is a
secret and secrets never belong in a persisted plan.

Slots understood in case content, door headers and door bodies:

``{{canary}}``
    The replica's marker.
``{{payload}}``
    The case content, verbatim.
``{{env:VAR}}``
    An environment variable, read when the door is called.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from norn.targets.manifest import ROLES, TargetManifest

#: Slots that carry a value the runner owns.
VALUE_SLOTS: frozenset[str] = frozenset({"canary", "payload"})

_SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?)\s*\}\}")


@dataclass(frozen=True)
class Plant:
    """One door call to make before the chat turn, with templates intact."""

    door_id: str
    method: str
    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Binding:
    """Result of binding one case variant to the manifest.

    ``reason`` is ``None`` when the case can run. Otherwise it names why the
    case cannot be mounted (``no_door:<id>``, ``no_role:<role>``,
    ``bad_slot:<name>``) and the case must be left out of the plan.
    """

    plants: tuple[Plant, ...] = ()
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.reason is None


def slots_in(value: Any) -> set[str]:
    """Collect every slot name appearing anywhere inside ``value``."""
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_SLOT_RE.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            found |= slots_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found |= slots_in(item)
    return found


def _known_slot(slot: str) -> bool:
    if slot in VALUE_SLOTS:
        return True
    prefix, _, name = slot.partition(":")
    return prefix == "env" and bool(name)


def bind_case(variant: Mapping[str, Any], manifest: TargetManifest) -> Binding:
    """Bind one corpus variant to the manifest's doors and roles.

    A plant is either a door id or a mapping ``{"door": id, "body": {...}}``.
    The mapping lets a case vary the envelope (subject, sender, one half of a
    fragment) without declaring a new door: an injection split across two
    emails is still the same door called twice.
    """
    raw_plant = variant.get("plant")
    if raw_plant is None:
        entries: tuple[Any, ...] = ()
    elif isinstance(raw_plant, (str, Mapping)):
        entries = (raw_plant,)
    else:
        entries = tuple(raw_plant)

    plants: list[Plant] = []
    for entry in entries:
        override: Mapping[str, Any] = {}
        if isinstance(entry, Mapping):
            door_id = str(entry.get("door") or "")
            raw_override = entry.get("body")
            override = raw_override if isinstance(raw_override, Mapping) else {}
        else:
            door_id = str(entry)
        door = manifest.door(door_id)
        if door is None:
            return Binding(reason=f"no_door:{door_id}")
        plants.append(
            Plant(
                door_id=door.id,
                method=door.method,
                endpoint=door.endpoint,
                headers=dict(door.headers),
                body={**door.body, **override},
            )
        )

    role = variant.get("requires_role")
    if role is not None:
        role = str(role)
        if role not in ROLES:
            return Binding(reason=f"no_role:{role}")
        if role not in set(manifest.roles().values()):
            return Binding(reason=f"no_role:{role}")

    for slot in sorted(slots_in(variant.get("content", "")) | slots_in(plants)):
        if not _known_slot(slot):
            return Binding(reason=f"bad_slot:{slot}")

    return Binding(plants=tuple(plants))


def render(value: Any, *, canary: str, payload: str) -> Any:
    """Resolve slots in ``value`` against the replica's canary and payload.

    A string that starts with ``b64:`` is a binary template: it is decoded, the
    slots are resolved inside it and the result is encoded back, because a PDF
    cannot carry ``{{payload}}`` and a pipeline that reads the metadata of one
    is exactly what some of these cases probe.
    """
    if isinstance(value, str):
        if value.startswith("b64:"):
            try:
                raw = base64.b64decode(value[4:], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("b64: template is not valid base64") from exc
            inner = render(raw.decode("utf-8", "replace"), canary=canary, payload=payload)
            return base64.b64encode(inner.encode("utf-8")).decode("ascii")

        def _replace(match: re.Match[str]) -> str:
            slot = match.group(1)
            if slot == "canary":
                return canary
            if slot == "payload":
                return payload
            prefix, _, name = slot.partition(":")
            if prefix == "env" and name:
                found = os.environ.get(name)
                if found is None or found == "":
                    raise ValueError(f"environment variable {name} is not set")
                return found
            raise ValueError(f"unknown slot: {slot!r}")

        return _SLOT_RE.sub(_replace, value)
    if isinstance(value, Mapping):
        return {key: render(item, canary=canary, payload=payload) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, canary=canary, payload=payload) for item in value]
    return value


def plant_request(
    plant: Plant, *, canary: str, payload: str
) -> tuple[str, str, dict[str, str], dict[str, Any] | None]:
    """Render one plant into ``(method, endpoint, headers, json_body)``."""
    headers = render(plant.headers, canary=canary, payload=payload)
    body = render(plant.body, canary=canary, payload=payload)
    return plant.method, plant.endpoint, dict(headers), (dict(body) or None)
