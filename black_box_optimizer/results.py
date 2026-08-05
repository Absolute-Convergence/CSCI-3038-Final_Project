"""Immutable final optimization-result contracts and construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from black_box_optimizer.models import OptimizationContract
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.stop_policy import TerminationReason


ResultStatus = Literal[
    "completed",
    "no_eligible_trials",
    "cancelled",
    "failed",
]

_VALID_RESULT_STATUSES = (
    "completed",
    "no_eligible_trials",
    "cancelled",
    "failed",
)
_NORMAL_TERMINATION_REASONS = (
    "maximum_trials",
    "search_exhausted",
    "excessive_failures",
)


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


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Authoritative immutable outcome of one optimization run."""

    history: tuple[TrialRecord, ...]
    pareto_front: ParetoFront
    termination_reason: TerminationReason
    status: ResultStatus

    def __post_init__(self) -> None:
        history = tuple(self.history)
        if not all(isinstance(record, TrialRecord) for record in history):
            raise TypeError("history must contain only TrialRecord values")
        if not isinstance(self.pareto_front, ParetoFront):
            raise TypeError("pareto_front must be a ParetoFront")
        if self.status not in _VALID_RESULT_STATUSES:
            valid = sorted(_VALID_RESULT_STATUSES)
            raise ValueError(f"status must be one of {valid}")

        history_ids = tuple(record.trial_id for record in history)
        if len(set(history_ids)) != len(history_ids):
            raise ValueError("history cannot contain duplicate trial IDs")
        if any(record not in history for record in self.pareto_front.records):
            raise ValueError("ParetoFront records must come from history")

        self._validate_status_semantics()
        object.__setattr__(self, "history", history)

    def _validate_status_semantics(self) -> None:
        reason = self.termination_reason
        if self.status in ("completed", "no_eligible_trials"):
            if reason not in _NORMAL_TERMINATION_REASONS:
                raise ValueError(
                    "normal result status requires normal termination"
                )
        elif self.status == "cancelled" and reason != "user_cancelled":
            raise ValueError(
                "cancelled status requires user_cancelled termination"
            )
        elif self.status == "failed" and reason != "fatal_error":
            raise ValueError("failed status requires fatal_error termination")

        if self.status == "completed" and not self.pareto_front.records:
            raise ValueError("completed status requires a nonempty ParetoFront")
        if (
            self.status == "no_eligible_trials"
            and self.pareto_front.records
        ):
            raise ValueError(
                "no_eligible_trials status requires an empty ParetoFront"
            )

    @property
    def attempted_count(self) -> int:
        return len(self.history)

    @property
    def successful_count(self) -> int:
        return sum(record.execution_succeeded for record in self.history)

    @property
    def valid_metrics_count(self) -> int:
        return sum(
            record.metrics_status == "valid" for record in self.history
        )

    @property
    def pareto_count(self) -> int:
        return len(self.pareto_front)


def build_optimization_result(
    history: Sequence[TrialRecord],
    contract: OptimizationContract,
    termination_reason: TerminationReason,
) -> OptimizationResult:
    """Derive the Pareto front and final status from authoritative evidence."""
    from black_box_optimizer.pareto import build_pareto_front

    history_snapshot = tuple(history)
    pareto_front = build_pareto_front(history_snapshot, contract)

    if termination_reason == "user_cancelled":
        status: ResultStatus = "cancelled"
    elif termination_reason == "fatal_error":
        status = "failed"
    elif pareto_front.records:
        status = "completed"
    else:
        status = "no_eligible_trials"

    return OptimizationResult(
        history=history_snapshot,
        pareto_front=pareto_front,
        termination_reason=termination_reason,
        status=status,
    )
