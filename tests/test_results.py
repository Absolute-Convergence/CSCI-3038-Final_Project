"""Focused tests for immutable final optimization results."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.results import (
    OptimizationResult,
    ParetoFront,
    build_optimization_result,
)


def make_contract() -> OptimizationContract:
    return OptimizationContract(
        parameters=(
            ParameterDefinition("x", ParameterKind.INTEGER, 0, 10),
        ),
        objectives=(
            Objective("score", Direction.MAXIMIZE),
            Objective("cost", Direction.MINIMIZE),
        ),
    )


def make_record(
    trial_id: int,
    *,
    score: float = 0.8,
    cost: float = 0.2,
    execution_status: str = "completed",
    metrics_status: str = "valid",
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parameters={"x": trial_id},
        metrics={"score": score, "cost": cost},
        execution_status=execution_status,
        metrics_status=metrics_status,
        runtime_seconds=0.1,
        exit_code=0 if execution_status == "completed" else 1,
        timed_out=False,
    )


class BuildOptimizationResultTests(unittest.TestCase):
    def test_normal_run_with_front_is_completed(self) -> None:
        records = (
            make_record(0, score=0.7, cost=0.4),
            make_record(1, score=0.9, cost=0.2),
        )

        result = build_optimization_result(
            records,
            make_contract(),
            "maximum_trials",
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "maximum_trials")
        self.assertEqual(
            tuple(record.trial_id for record in result.pareto_front),
            (1,),
        )

    def test_normal_run_without_eligible_records_is_explicit(self) -> None:
        failed = make_record(
            0,
            execution_status="process_failed",
            metrics_status="missing",
        )

        result = build_optimization_result(
            (failed,),
            make_contract(),
            "maximum_trials",
        )

        self.assertEqual(result.status, "no_eligible_trials")
        self.assertEqual(result.pareto_front.records, ())

    def test_cancelled_result_keeps_partial_valid_front(self) -> None:
        eligible = make_record(0)
        cancelled = TrialRecord(
            trial_id=1,
            parameters={"x": 1},
            metrics={},
            execution_status="cancelled",
            metrics_status="missing",
            runtime_seconds=0.2,
            exit_code=None,
            timed_out=False,
        )

        result = build_optimization_result(
            (eligible, cancelled),
            make_contract(),
            "user_cancelled",
        )

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.pareto_front.records, (eligible,))

    def test_failed_result_keeps_partial_valid_front(self) -> None:
        eligible = make_record(0)

        result = build_optimization_result(
            (eligible,),
            make_contract(),
            "fatal_error",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.pareto_front.records, (eligible,))

    def test_counts_are_derived_from_history_and_front(self) -> None:
        completed = make_record(0)
        failed = make_record(
            1,
            execution_status="process_failed",
            metrics_status="missing",
        )
        result = build_optimization_result(
            (completed, failed),
            make_contract(),
            "maximum_trials",
        )

        self.assertEqual(result.attempted_count, 2)
        self.assertEqual(result.successful_count, 1)
        self.assertEqual(result.valid_metrics_count, 1)
        self.assertEqual(result.pareto_count, 1)


class OptimizationResultValidationTests(unittest.TestCase):
    def test_history_is_copied_and_result_is_frozen(self) -> None:
        record = make_record(0)
        history = [record]
        result = OptimizationResult(
            history=history,
            pareto_front=ParetoFront((record,)),
            termination_reason="maximum_trials",
            status="completed",
        )
        history.clear()

        self.assertEqual(result.history, (record,))
        with self.assertRaises(FrozenInstanceError):
            result.status = "failed"

    def test_front_record_must_come_from_history(self) -> None:
        history_record = make_record(0)
        other_record = make_record(1)

        with self.assertRaisesRegex(ValueError, "come from history"):
            OptimizationResult(
                history=(history_record,),
                pareto_front=ParetoFront((other_record,)),
                termination_reason="maximum_trials",
                status="completed",
            )

    def test_status_must_match_termination_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "user_cancelled"):
            OptimizationResult(
                history=(),
                pareto_front=ParetoFront(()),
                termination_reason="fatal_error",
                status="cancelled",
            )

    def test_completed_status_requires_front(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty"):
            OptimizationResult(
                history=(),
                pareto_front=ParetoFront(()),
                termination_reason="search_exhausted",
                status="completed",
            )

    def test_history_rejects_non_trial_record_items(self) -> None:
        with self.assertRaisesRegex(TypeError, "TrialRecord"):
            OptimizationResult(
                history=("not a record",),
                pareto_front=ParetoFront(()),
                termination_reason="maximum_trials",
                status="no_eligible_trials",
            )

    def test_pareto_front_must_be_a_pareto_front_instance(self) -> None:
        with self.assertRaisesRegex(TypeError, "ParetoFront"):
            OptimizationResult(
                history=(),
                pareto_front=(),
                termination_reason="maximum_trials",
                status="no_eligible_trials",
            )

    def test_status_must_be_a_known_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must be one of"):
            OptimizationResult(
                history=(),
                pareto_front=ParetoFront(()),
                termination_reason="maximum_trials",
                status="bogus_status",
            )

    def test_history_cannot_contain_duplicate_trial_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate trial IDs"):
            OptimizationResult(
                history=(make_record(0), make_record(0)),
                pareto_front=ParetoFront(()),
                termination_reason="maximum_trials",
                status="completed",
            )

    def test_completed_status_rejects_an_abnormal_termination_reason(
        self,
    ) -> None:
        # "completed" is only ever reached via maximum_trials or
        # search_exhausted -- any other reason paired with it is invalid
        # regardless of what the Pareto front looks like.
        with self.assertRaisesRegex(ValueError, "normal termination"):
            OptimizationResult(
                history=(),
                pareto_front=ParetoFront(()),
                termination_reason="user_cancelled",
                status="completed",
            )

    def test_failed_status_requires_fatal_error_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "fatal_error termination"):
            OptimizationResult(
                history=(),
                pareto_front=ParetoFront(()),
                termination_reason="user_cancelled",
                status="failed",
            )

    def test_no_eligible_trials_status_requires_an_empty_front(self) -> None:
        record = make_record(0)
        with self.assertRaisesRegex(ValueError, "empty ParetoFront"):
            OptimizationResult(
                history=(record,),
                pareto_front=ParetoFront((record,)),
                termination_reason="maximum_trials",
                status="no_eligible_trials",
            )


if __name__ == "__main__":
    unittest.main()
