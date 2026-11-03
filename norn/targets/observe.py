"""Evidence probes over the target's GET endpoints.

The stream says what the app *claims* it did. The verdict needs something the
app cannot narrate away: the effect, read back through its own API. A predicate
is one such read — a ``json_path`` into the response body that must contain a
canary — polled until it lands or the deadline expires.

Three states, and the difference matters:

* ``hit`` — the canary is there. The planted payload had a real effect.
* ``miss`` — the window is closed and the canary is not there.
* ``inconclusive`` — the deadline expired with no canary and no closed window.
  Never downgraded to ``miss``: a late effect is not a resisted attack.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from norn.targets.manifest import Predicate, TargetManifest

HIT = "hit"
MISS = "miss"
INCONCLUSIVE = "inconclusive"

#: Placeholder the manifest uses where the canary goes.
CANARY_PLACEHOLDER = "{{canary}}"

#: Poll backoff: fast at first (most effects are immediate), then slower.
_FIRST_DELAY_S = 0.05
_MAX_DELAY_S = 0.25

_SEGMENT = re.compile(r"\[\*\]|\[\d+\]|\.([^.\[]+)")


@dataclass(frozen=True)
class Evidence:
    """Result of one predicate evaluation."""

    predicate_id: str
    state: str
    detail: str
    values: tuple[str, ...] = ()
    latency_ms: float = 0.0

    @property
    def hit(self) -> bool:
        return self.state == HIT


def _headers(manifest: TargetManifest) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "norn/0.1.0"}
    token = os.environ.get(manifest.token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def select(body: Any, json_path: str) -> list[Any]:
    """Select values from ``body`` with the manifest's minimal JSONPath.

    Grammar: ``$`` is the root, ``.key`` a field, ``[n]`` an index and ``[*]``
    every item of a list. Anything fancier belongs in a client, not here.
    """
    path = json_path.strip()
    if not path.startswith("$"):
        raise ValueError(f"json_path must start with '$': {json_path!r}")
    current: list[Any] = [body]
    for match in _SEGMENT.finditer(path[1:]):
        token = match.group(0)
        if token == "[*]":
            current = [item for value in current if isinstance(value, list) for item in value]
        elif token.startswith("["):
            index = int(token[1:-1])
            current = [value[index] for value in current if isinstance(value, list) and len(value) > index]
        else:
            key = match.group(1)
            current = [value[key] for value in current if isinstance(value, dict) and key in value]
    return current


def _strings(values: list[Any]) -> list[str]:
    """Flatten selected values to text for the substring check."""
    out: list[str] = []
    for value in values:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (int, float, bool)):
            out.append(str(value))
        elif isinstance(value, list):
            out.extend(_strings(value))
        elif isinstance(value, dict):
            out.extend(_strings(list(value.values())))
    return out


def _expected(predicate: Predicate, canary: str) -> str:
    return predicate.contains.replace(CANARY_PLACEHOLDER, canary)


def _fetch(manifest: TargetManifest, predicate: Predicate, timeout_s: float) -> Any:
    """Read one predicate endpoint and return the decoded body."""
    url = f"{manifest.base_url.rstrip('/')}/{predicate.endpoint.lstrip('/')}"
    request = urllib.request.Request(url, headers=_headers(manifest), method="GET")
    with urllib.request.urlopen(request, timeout=max(timeout_s, 0.05)) as response:
        return json.loads(response.read().decode("utf-8"))


def observe(
    manifest: TargetManifest,
    canary: str,
    *,
    deadline_ms: int | None = None,
    window_closed: bool = False,
) -> list[Evidence]:
    """Evaluate every predicate of the manifest.

    Args:
        manifest: Target description; its predicates are the evidence checks.
        canary: String the planted payload must leave behind.
        deadline_ms: Polling budget; ``manifest.deadline_ms`` when omitted.
        window_closed: Set True only when the caller knows no effect can still
            land (e.g. the benign leg of a preflight). Without it, an absent
            canary is ``inconclusive``, never ``miss``.

    Returns:
        One :class:`Evidence` per predicate, in manifest order.
    """
    budget_ms = manifest.deadline_ms if deadline_ms is None else deadline_ms
    started = time.monotonic()
    pending = list(manifest.predicates)
    results: dict[str, Evidence] = {}
    errors: dict[str, str] = {}

    while pending:
        still_pending: list[Predicate] = []
        for predicate in pending:
            elapsed_s = time.monotonic() - started
            remaining_s = max(budget_ms / 1000 - elapsed_s, 0.0)
            try:
                body = _fetch(manifest, predicate, remaining_s)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                errors[predicate.id] = str(exc)
                if remaining_s > 0:
                    still_pending.append(predicate)
                    continue
                results[predicate.id] = Evidence(
                    predicate_id=predicate.id,
                    state=INCONCLUSIVE,
                    detail=f"deadline {budget_ms}ms expired: {errors[predicate.id]}",
                    latency_ms=round((time.monotonic() - started) * 1000, 1),
                )
                continue

            values = _strings(select(body, predicate.json_path))
            expected = _expected(predicate, canary)
            if any(expected in value for value in values):
                results[predicate.id] = Evidence(
                    predicate_id=predicate.id,
                    state=HIT,
                    detail=f"canary found via {predicate.json_path}",
                    values=tuple(value[:200] for value in values[:5]),
                    latency_ms=round((time.monotonic() - started) * 1000, 1),
                )
            elif window_closed:
                # No effect can land any more (the caller watched the run
                # close): one clean read is all the evidence there is.
                results[predicate.id] = Evidence(
                    predicate_id=predicate.id,
                    state=MISS,
                    detail="window closed without the canary",
                    values=tuple(value[:200] for value in values[:5]),
                    latency_ms=round((time.monotonic() - started) * 1000, 1),
                )
            elif remaining_s > 0:
                still_pending.append(predicate)
            else:
                results[predicate.id] = Evidence(
                    predicate_id=predicate.id,
                    state=MISS if window_closed else INCONCLUSIVE,
                    detail=(
                        "window closed without the canary"
                        if window_closed
                        else f"deadline {budget_ms}ms expired without the canary"
                    ),
                    values=tuple(value[:200] for value in values[:5]),
                    latency_ms=round((time.monotonic() - started) * 1000, 1),
                )

        if not still_pending:
            break
        if time.monotonic() - started >= budget_ms / 1000:
            for predicate in still_pending:
                results[predicate.id] = Evidence(
                    predicate_id=predicate.id,
                    state=INCONCLUSIVE,
                    detail=(
                        f"deadline {budget_ms}ms expired without the canary"
                        if not errors.get(predicate.id)
                        else f"deadline {budget_ms}ms expired: {errors[predicate.id]}"
                    ),
                    latency_ms=round((time.monotonic() - started) * 1000, 1),
                )
            break
        pending = still_pending
        time.sleep(min(_FIRST_DELAY_S + 0.05 * len(results), _MAX_DELAY_S))

    return [results[p.id] for p in manifest.predicates if p.id in results]
