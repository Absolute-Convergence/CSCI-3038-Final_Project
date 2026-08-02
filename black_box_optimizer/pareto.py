"""Mixed-direction Pareto eligibility, dominance, and front evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from black_box_optimizer.models import Direction, OptimizationContract
from black_box_optimizer.records import TrialRecord


@dataclass(frozen=True, slots=True)
class ParetoFront:
    """Immutable, history-ordered non-dominated trial records."""

    records: tuple[TrialRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not all(isinstance(record, TrialRecord) for record in records):
            raise TypeError("records must contain only TrialRecord values")
        trial_ids = tuple(record.trial_id for record in records)
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("ParetoFront cannot contain duplicate trial IDs")
        object.__setattr__(self, "records", records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def is_eligible(
    record: TrialRecord,
    contract: OptimizationContract,
) -> bool:
    """Return whether a record has every fact needed for Pareto evaluation."""
    if not record.execution_succeeded:
        return False
    if record.metrics_status != "valid":
        return False

    return all(
        objective.metric_name in record.metrics
        and math.isfinite(record.metrics[objective.metric_name])
        for objective in contract.objectives
    )


def dominates(
    left: TrialRecord,
    right: TrialRecord,
    contract: OptimizationContract,
) -> bool:
    """Return whether left is no worse everywhere and better somewhere."""
    if not is_eligible(left, contract) or not is_eligible(right, contract):
        return False

    strictly_better = False
    for objective in contract.objectives:
        left_value = left.metrics[objective.metric_name]
        right_value = right.metrics[objective.metric_name]

        if objective.direction is Direction.MINIMIZE:
            if left_value > right_value:
                return False
            strictly_better = strictly_better or left_value < right_value
        else:
            if left_value < right_value:
                return False
            strictly_better = strictly_better or left_value > right_value

    return strictly_better


def build_pareto_front(
    records: Sequence[TrialRecord],
    contract: OptimizationContract,
) -> ParetoFront:
    """Return every eligible record not dominated by another eligible record."""
    eligible = tuple(
        record for record in records if is_eligible(record, contract)
    )
    non_dominated = tuple(
        candidate
        for candidate in eligible
        if not any(
            challenger is not candidate
            and dominates(challenger, candidate, contract)
            for challenger in eligible
        )
    )
    return ParetoFront(records=non_dominated)
