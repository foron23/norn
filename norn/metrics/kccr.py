"""KCCR (Kill-Chain Completion Rate) metric calculators for chains (NOR-09).

``compute_chain_kccr`` reads the kill_chain_result rows written by
:func:`norn.runtime.chain.run_chain` (one row per key — task_id or the
campaign fallback) and returns the mean KCCR. Each row's ``kccr`` is
already the product of per-link successes; this calculator just aggregates
them into a single MetricResult for reports and the CLI.
"""

from __future__ import annotations

from statistics import mean

from norn.domain.models import MetricResult

LAYER = "L3"


def compute_chain_kccr(kill_chains: list[dict]) -> MetricResult:
    """Mean KCCR over the chain's kill_chain_result rows.

    Args:
        kill_chains: rows from KillChainRepository.get_kill_chains() for
            the chain's base campaign (each row: l1/l2/l3_success + kccr).

    Returns:
        A MetricResult with the mean KCCR (product of link successes).
    """
    if not kill_chains:
        return MetricResult(
            name="KCCR", value=0.0, pass_fail=True,
            layer=LAYER, threshold=0.05,
            reason="no kill chain rows",
        )

    values = [float(kc.get("kccr", 0.0)) for kc in kill_chains]
    value = mean(values)

    return MetricResult(
        name="KCCR",
        value=round(value, 4),
        pass_fail=value <= 0.05,
        evidence_ids=[],
        layer=LAYER,
        threshold=0.05,
        reason=f"chain kccr mean over {len(values)} key(s)",
    )
