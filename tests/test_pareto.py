"""
Focused tests for is_eligible() from pareto.py.

Right now pareto.py only has the eligibility check because persistence
needs it for history.csv. These tests cover that function through the
real TrialRecord and OptimizationContract implementations.

KNOWN GAP!!! Once the dominance comparison and full ParetoFront sweep
are added, this file will need more tests for those pieces too. For now
it is intentionally just the is_eligible() corner of the Pareto world.
"""

from __future__ import annotations

import unittest

from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.records import TrialRecord


def make_contract() -> OptimizationContract:
    """Build the same two-objective contract used throughout these tests."""
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.01, 0.1
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


def make_record(
    execution_status: str = "completed",
    metrics_status: str = "valid",
    metrics: dict[str, float] | None = None,
) -> TrialRecord:
    """Build a small TrialRecord with only the fields these tests need."""
    return TrialRecord(
        trial_id=0,
        parameters={"learning_rate": 0.05},
        metrics={} if metrics is None else metrics,
        execution_status=execution_status,
        metrics_status=metrics_status,
        runtime_seconds=1.0,
        exit_code=0 if execution_status == "completed" else 1,
        timed_out=False,
    )


class IsEligibleTests(unittest.TestCase):
    """Every way a TrialRecord can pass or fail Pareto eligibility."""

    def test_eligible_when_execution_succeeded_and_metrics_valid(self) -> None:
        contract = make_contract()
        record = make_record(metrics={"accuracy": 0.9, "loss": 0.1})

        self.assertTrue(is_eligible(record, contract))

    def test_extra_undeclared_metric_does_not_affect_eligibility(self) -> None:
        contract = make_contract()
        record = make_record(
            metrics={
                "accuracy": 0.9,
                "loss": 0.1,
                "training_time_seconds": 3.2,
            }
        )

        self.assertTrue(is_eligible(record, contract))

    def test_not_eligible_when_execution_did_not_succeed(self) -> None:
        contract = make_contract()
        record = make_record(
            execution_status="process_failed",
            metrics_status="missing",
            metrics={},
        )

        self.assertFalse(is_eligible(record, contract))

    def test_execution_check_is_independent_of_metrics_status(self) -> None:
        # execution_succeeded must be checked on its own, not just as a
        # side effect of metrics_status usually being invalid too --
        # TrialRecord doesn't couple these two fields together, so build
        # one directly with a "valid" metrics_status despite the failed
        # execution to prove the two checks are genuinely independent.
        contract = make_contract()
        record = make_record(
            execution_status="process_failed",
            metrics_status="valid",
            metrics={"accuracy": 0.9, "loss": 0.1},
        )

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_metrics_status_missing(self) -> None:
        contract = make_contract()
        record = make_record(metrics_status="missing", metrics={})

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_metrics_status_malformed(self) -> None:
        contract = make_contract()
        record = make_record(metrics_status="malformed", metrics={})

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_metrics_status_nonfinite(self) -> None:
        contract = make_contract()
        record = make_record(metrics_status="nonfinite", metrics={})

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_one_objective_metric_is_missing(self) -> None:
        contract = make_contract()

        # accuracy exists but loss is fully missing
        record = make_record(metrics={"accuracy": 0.9})

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_an_objective_value_is_infinite(self) -> None:
        contract = make_contract()
        record = make_record(
            metrics={"accuracy": 0.9, "loss": float("inf")}
        )

        self.assertFalse(is_eligible(record, contract))

    def test_not_eligible_when_an_objective_value_is_nan(self) -> None:
        # NaN and infinite both fail math.isfinite(), but they're
        # different failure modes (a computation error vs. an
        # overflow), so both are worth covering on their own.
        contract = make_contract()
        record = make_record(
            metrics={"accuracy": 0.9, "loss": float("nan")}
        )

        self.assertFalse(is_eligible(record, contract))

    def test_does_not_mutate_the_record(self) -> None:
        contract = make_contract()
        record = make_record(metrics={"accuracy": 0.9, "loss": 0.1})
        before = dict(record.metrics)

        is_eligible(record, contract)

        self.assertEqual(dict(record.metrics), before)


if __name__ == "__main__":
    unittest.main()