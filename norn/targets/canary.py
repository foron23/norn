"""Per-replica canary markers.

A canary is the only thing a case has to leave behind for the evidence
predicates to be able to say "the effect happened". It is derived, not random:
the same case and replica produce the same marker, so re-reading a campaign
rebuilds the string that was planted.
"""

from __future__ import annotations

import hashlib
import re

#: Prefix shared with the canary corpus and ``CanaryScorer``.
PREFIX = "CNRY-"

#: Shape of a marker: prefix plus eight hex characters.
PATTERN = re.compile(r"CNRY-[0-9a-f]{8}")


def make_canary(*parts: object) -> str:
    """Return a deterministic marker for the given parts."""
    seed = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{PREFIX}{digest[:8]}"
