"""
pareto.py

Multi objective eligibility and Pareto evaluation.

Only is_eligible() is implemented so far because it's
dinky small, fully self-contained, and given as near literal
pseudocode in the docs, so I implemented exactly as written
rather than approximating! Feel free to tweak if u need
(I wanted to finish persistence)

KNOWN GAP!!! The dominance comparison (section 8.2) and the ParetoFront
sweep algorithm (section 8.3) are not implemented. This file
intentionally leaves that seam open rather than faking a ParetoFront or
inventing a weighted score see controller.py's FINALIZING seam,
which is blocked on this same gap.

Also!! Mel when you finish this, I only wrote the tests
that have to do with is_eligible() so you'll need to update
test_pareto as well!
"""

from __future__ import annotations

import math

from black_box_optimizer.models import OptimizationContract
from black_box_optimizer.records import TrialRecord


def is_eligible(
    record: TrialRecord, contract: OptimizationContract
) -> bool:
    """
    Whether one TrialRecord is eligible for Pareto consideration.

    Requires a successfully executed trial with valid metrics that
    include every declared objective, all as finite values. An extra
    metric that isn't declared as an objective has no effect.
    """
    if not record.execution_succeeded:
        return False
    if record.metrics_status != "valid":
        return False
    return all(
        objective.metric_name in record.metrics
        and math.isfinite(record.metrics[objective.metric_name])
        for objective in contract.objectives
    )
