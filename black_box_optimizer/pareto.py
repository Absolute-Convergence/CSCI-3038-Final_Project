"""
pareto.py

Multi-objective eligibility and Pareto evaluation.

Right now this file only implements is_eligible(). It's dinky,
completely self-contained, and the TDS basically hands over the logic as
pseudocode, so I just implemented it directly instead of getting fancy.

Keep an eye out (!!!) for the KNOWN GAP note below. The rest of the
Pareto implementation is intentionally left unfinished while the project
waits on the full ParetoFront sweep.

Also!! Mel, when you finish this I only wrote the tests for
is_eligible(), so test_pareto.py will need some love too :)
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
    include every declared objective, all as finite values. Extra
    metrics are totally fine. They just don't matter for Pareto
    eligibility.
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


# KNOWN GAP
# The dominance comparison and full ParetoFront sweep from TDS section 8
# are intentionally left unimplemented
# controller.py FINALIZING is waiting on these pieces before it can build
# the OptimizationResult and finish reporting