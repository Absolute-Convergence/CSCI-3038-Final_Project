"""Focused tests for mixed-direction Pareto evaluation."""

from __future__ import annotations

import unittest

from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.pareto import (
    ParetoFront,
    build_pareto_front,
    dominates,
    is_eligible,
)
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
    trial_id: int = 0,
    execution_status: str = "completed",
    metrics_status: str = "valid",
    metrics: dict[str, float] | None = None,
) -> TrialRecord:
    """Build a small TrialRecord with only the fields these tests need."""
    return TrialRecord(
        trial_id=trial_id,
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


class DominanceTests(unittest.TestCase):
    def test_mixed_direction_dominance(self) -> None:
        contract = make_contract()
        better = make_record(metrics={"accuracy": 0.9, "loss": 0.2})
        worse = make_record(
            trial_id=1,
            metrics={"accuracy": 0.8, "loss": 0.3},
        )

        self.assertTrue(dominates(better, worse, contract))
        self.assertFalse(dominates(worse, better, contract))

    def test_tradeoff_does_not_dominate(self) -> None:
        contract = make_contract()
        accurate = make_record(metrics={"accuracy": 0.9, "loss": 0.4})
        low_loss = make_record(
            trial_id=1,
            metrics={"accuracy": 0.8, "loss": 0.2},
        )

        self.assertFalse(dominates(accurate, low_loss, contract))
        self.assertFalse(dominates(low_loss, accurate, contract))

    def test_equal_objectives_do_not_strictly_dominate(self) -> None:
        contract = make_contract()
        left = make_record(metrics={"accuracy": 0.9, "loss": 0.2})
        right = make_record(
            trial_id=1,
            metrics={"accuracy": 0.9, "loss": 0.2},
        )

        self.assertFalse(dominates(left, right, contract))
        self.assertFalse(dominates(right, left, contract))

    def test_ineligible_record_never_dominates(self) -> None:
        contract = make_contract()
        failed = make_record(
            execution_status="process_failed",
            metrics_status="valid",
            metrics={"accuracy": 1.0, "loss": 0.0},
        )
        completed = make_record(
            trial_id=1,
            metrics={"accuracy": 0.5, "loss": 0.5},
        )

        self.assertFalse(dominates(failed, completed, contract))


class ParetoFrontTests(unittest.TestCase):
    def test_hand_calculated_mixed_direction_front(self) -> None:
        contract = make_contract()
        records = (
            make_record(
                trial_id=0,
                metrics={"accuracy": 0.8, "loss": 0.4},
            ),
            make_record(
                trial_id=1,
                metrics={"accuracy": 0.85, "loss": 0.35},
            ),
            make_record(
                trial_id=2,
                metrics={"accuracy": 0.9, "loss": 0.5},
            ),
            make_record(
                trial_id=3,
                metrics={"accuracy": 0.75, "loss": 0.3},
            ),
            make_record(
                trial_id=4,
                metrics={"accuracy": 0.85, "loss": 0.35},
            ),
            make_record(
                trial_id=5,
                execution_status="process_failed",
                metrics_status="valid",
                metrics={"accuracy": 1.0, "loss": 0.0},
            ),
        )

        front = build_pareto_front(records, contract)

        self.assertEqual(
            tuple(record.trial_id for record in front.records),
            (1, 2, 3, 4),
        )

    def test_empty_eligible_set_returns_empty_front(self) -> None:
        failed = make_record(
            execution_status="process_failed",
            metrics_status="missing",
        )

        front = build_pareto_front((failed,), make_contract())

        self.assertEqual(front.records, ())

    def test_front_copies_input_to_tuple(self) -> None:
        records = [
            make_record(metrics={"accuracy": 0.9, "loss": 0.2})
        ]

        front = ParetoFront(records=records)
        records.clear()

        self.assertEqual(len(front), 1)

    def test_front_rejects_duplicate_trial_ids(self) -> None:
        first = make_record(metrics={"accuracy": 0.9, "loss": 0.2})
        duplicate = make_record(metrics={"accuracy": 0.8, "loss": 0.3})

        with self.assertRaisesRegex(ValueError, "duplicate trial IDs"):
            ParetoFront(records=(first, duplicate))


if __name__ == "__main__":
    unittest.main()
