"""HTTP/SSE target connectors.

A target is a real application measured through its own user-facing door: one
chat endpoint streaming Server-Sent Events plus the GET endpoints that carry
evidence of the effect. :mod:`norn.targets.manifest` defines and validates the
manifest that describes it.
"""

from __future__ import annotations

from norn.targets.manifest import (
    ALLOWED_METHODS,
    CANONICAL_EVENTS,
    CAPABILITIES,
    Predicate,
    TargetManifest,
    from_dict,
    load_manifest,
)

__all__ = [
    "ALLOWED_METHODS",
    "CANONICAL_EVENTS",
    "CAPABILITIES",
    "Predicate",
    "TargetManifest",
    "from_dict",
    "load_manifest",
]
