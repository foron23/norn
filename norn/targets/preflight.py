"""Target preflight: prove the oracle can tell an effect from a clean app.

A green campaign against a target that cannot see anything is worse than a red
one: it reports "the app resisted" when the truth is "the probe was blind".
Preflight answers that question before any campaign runs, with the same two
doors a campaign uses:

* **negative** — read the evidence endpoints *before* planting anything. The
  predicates must say no. If they say yes, they match everything and the oracle
  is blind.
* **positive** — drive one chat turn carrying the canary and read again. The
  predicates must now say yes.

Only when the same predicate answers *no* and then *yes* is the channel proven.
A stream that never closes is reported separately (``stream_unfinished``), since
a buffered app looks exactly like a resistant one from the outside.

CLI::

    python -m norn.targets.preflight --manifest tests/fixtures/fake.yaml
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from norn.targets.client import RunRecord, TargetError, chat
from norn.targets.manifest import TargetManifest, load_manifest
from norn.targets.observe import HIT, Evidence, observe

#: Verdicts preflight can return.
OK = "ok"
ORACLE_BLIND = "oracle_blind"
STREAM_UNFINISHED = "stream_unfinished"


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of the two-leg probe."""

    status: str
    canary: str
    negative: tuple[Evidence, ...] = ()
    positive: tuple[Evidence, ...] = ()
    chat: RunRecord | None = None
    detail: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def summary(self) -> str:
        """One-line verdict, e.g. ``OK positive=hit negative=miss``."""
        negative = {e.state for e in self.negative} or {"n/a"}
        positive = {e.state for e in self.positive} or {"n/a"}
        prefix = "OK" if self.ok else self.status.upper()
        return f"{prefix} positive={'/'.join(sorted(positive))} negative={'/'.join(sorted(negative))}"


def preflight(manifest: TargetManifest, *, canary: str | None = None) -> PreflightResult:
    """Run the negative and positive legs against ``manifest``.

    Args:
        manifest: Target to check.
        canary: Marker to plant; generated when omitted.

    Returns:
        A :class:`PreflightResult`. ``status`` is ``ok`` only when the negative
        leg misses and the positive leg hits.

    Raises:
        TargetError: The chat endpoint could not be reached at all.
    """
    marker = canary or f"NORN-PREFLIGHT-{uuid.uuid4().hex[:12]}"

    # Leg 1 — nothing planted yet: every predicate must say no.
    negative = tuple(observe(manifest, marker, window_closed=True))

    # Leg 2 — one real chat turn carrying the marker.
    try:
        record = chat(manifest, f"Availability check for account {marker}.")
    except TargetError as exc:
        if "cannot reach target" in str(exc):
            raise
        return PreflightResult(
            status=STREAM_UNFINISHED,
            canary=marker,
            negative=negative,
            detail=str(exc),
            errors=(str(exc),),
        )

    if not record.finished:
        return PreflightResult(
            status=STREAM_UNFINISHED,
            canary=marker,
            negative=negative,
            chat=record,
            detail=(
                f"stream closed without run.finished within {manifest.deadline_ms}ms "
                "(a buffered or truncated stream is not a stream)"
            ),
            errors=record.violations,
        )

    positive = tuple(observe(manifest, marker))

    false_positives = [e.predicate_id for e in negative if e.hit]
    if false_positives:
        return PreflightResult(
            status=ORACLE_BLIND,
            canary=marker,
            negative=negative,
            positive=positive,
            chat=record,
            detail=(
                "predicate(s) matched before anything was planted: "
                f"{', '.join(false_positives)}"
            ),
        )

    missing = [e.predicate_id for e in positive if e.state != HIT]
    if missing:
        states = {e.predicate_id: e.state for e in positive if e.predicate_id in missing}
        return PreflightResult(
            status=ORACLE_BLIND,
            canary=marker,
            negative=negative,
            positive=positive,
            chat=record,
            detail=(
                "the planted canary produced no evidence: "
                + ", ".join(f"{pid}={state}" for pid, state in states.items())
            ),
        )

    return PreflightResult(
        status=OK,
        canary=marker,
        negative=negative,
        positive=positive,
        chat=record,
        detail="the same predicates answered no before planting and yes after",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: exit 0 only when the target passes preflight."""
    parser = argparse.ArgumentParser(
        prog="norn.targets.preflight",
        description="Check that a target's evidence channel can tell an effect from a clean run.",
    )
    parser.add_argument("--manifest", required=True, help="path to the target manifest YAML")
    parser.add_argument("--canary", default=None, help="marker to plant (generated by default)")
    parser.add_argument("--quiet", action="store_true", help="only print the one-line verdict")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)

    try:
        result = preflight(manifest, canary=args.canary)
    except TargetError as exc:
        print(f"ERROR unreachable: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"target: {manifest.name} ({manifest.base_url})")
        print(f"capabilities: {', '.join(sorted(manifest.capabilities))}")
        print(f"canary: {result.canary}")
        for evidence in (*result.negative, *result.positive):
            print(f"  {evidence.predicate_id}: {evidence.state} — {evidence.detail}")
        if result.detail:
            print(f"detail: {result.detail}")
    print(result.summary())

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
