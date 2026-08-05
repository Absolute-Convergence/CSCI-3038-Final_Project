"""Focused tests for StopDecision and StopPolicyEvaluator."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    StopPolicy,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.stop_policy import StopDecision, StopPolicyEvaluator


def make_trial_record(trial_id: int) -> TrialRecord:
    """Build a minimal, valid TrialRecord for filling out fake history."""
    return TrialRecord(
        trial_id=trial_id,
        parameters={"learning_rate": 0.01},
        metrics={"accuracy": 0.9, "loss": 0.1},
        execution_status="completed",
        metrics_status="valid",
        runtime_seconds=1.0,
        exit_code=0,
        timed_out=False,
    )


def make_failed_trial_record(trial_id: int) -> TrialRecord:
    """Build a TrialRecord that is_eligible() will reject."""
    return TrialRecord(
        trial_id=trial_id,
        parameters={"learning_rate": 0.01},
        metrics={},
        execution_status="process_failed",
        metrics_status="missing",
        runtime_seconds=0.1,
        exit_code=2,
        timed_out=False,
        error_message="Worker exited with code 2",
    )


def make_contract() -> OptimizationContract:
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.0001, 0.1
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


class StopDecisionTests(unittest.TestCase):
    """Verify construction, validation, and immutability of StopDecision."""

    # Tests that the decision and termination reason agree with each other.

    def test_continue_with_reason_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be True"):
            StopDecision(
                continue_execution=True, termination_reason="maximum_trials"
            )

    def test_stop_without_reason_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            StopDecision(continue_execution=False, termination_reason=None)

    # Tests that reject bad field values.

    def test_invalid_termination_reason_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "termination_reason"):
            StopDecision(continue_execution=False, termination_reason="banana")

    def test_non_bool_continue_execution_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "bool"):
            StopDecision(continue_execution="yes", termination_reason=None)

    def test_non_string_message_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "string"):
            StopDecision(
                continue_execution=True, termination_reason=None, message=404
            )

    def test_decision_is_frozen(self) -> None:
        decision = StopDecision(
            continue_execution=True, termination_reason=None
        )
        with self.assertRaises(FrozenInstanceError):
            decision.continue_execution = False  # type: ignore[misc]


class StopPolicyEvaluatorTests(unittest.TestCase):
    """Verify before_trial and after_trial against the trial count."""

    def setUp(self) -> None:
        self.evaluator = StopPolicyEvaluator(StopPolicy(max_trials=3))

    # Tests that check continuing while under the trial limit.

    def test_before_trial_continues_when_under_limit(self) -> None:
        history = [make_trial_record(1)]
        decision = self.evaluator.before_trial(history)

        self.assertTrue(decision.continue_execution)
        self.assertIsNone(decision.termination_reason)

    def test_after_trial_continues_when_under_limit(self) -> None:
        history = [make_trial_record(1)]
        decision = self.evaluator.after_trial(history)

        self.assertTrue(decision.continue_execution)
        self.assertIsNone(decision.termination_reason)

    def test_empty_history_continues(self) -> None:
        decision = self.evaluator.before_trial([])
        self.assertTrue(decision.continue_execution)

    # Tests that check stopping once the trial limit is reached.

    def test_before_trial_stops_at_limit(self) -> None:
        history = [make_trial_record(i) for i in range(3)]
        decision = self.evaluator.before_trial(history)

        self.assertFalse(decision.continue_execution)
        self.assertEqual(decision.termination_reason, "maximum_trials")
        self.assertEqual(decision.message, "Reached the maximum of 3 trials.")

    def test_after_trial_stops_at_limit(self) -> None:
        history = [make_trial_record(i) for i in range(3)]
        decision = self.evaluator.after_trial(history)

        self.assertFalse(decision.continue_execution)
        self.assertEqual(decision.termination_reason, "maximum_trials")
        self.assertEqual(decision.message, "Reached the maximum of 3 trials.")

    def test_stops_when_history_exceeds_limit(self) -> None:
        # Shouldn't normally happen, but the evaluator should still refuse
        # to continue rather than assume everything is fine.
        history = [make_trial_record(i) for i in range(5)]
        decision = self.evaluator.before_trial(history)

        self.assertFalse(decision.continue_execution)


class ExcessiveFailuresTests(unittest.TestCase):
    """Verify the repeated-failure early-stop check (contract supplied)."""

    def setUp(self) -> None:
        self.evaluator = StopPolicyEvaluator(
            StopPolicy(max_trials=150), make_contract()
        )

    def test_no_contract_never_triggers_regardless_of_failure_rate(
        self,
    ) -> None:
        # Backward-compat: every pre-existing caller constructs
        # StopPolicyEvaluator with just a StopPolicy, and must keep
        # behaving exactly as before -- max_trials only, no failure check.
        evaluator = StopPolicyEvaluator(StopPolicy(max_trials=150))
        history = [make_failed_trial_record(i) for i in range(10)]

        decision = evaluator.before_trial(history)

        self.assertTrue(decision.continue_execution)

    def test_five_consecutive_failures_does_not_trigger_yet(self) -> None:
        # Hand-verified: binomial_tail_probability(5, 5, 0.3) = 0.00243,
        # not below the calibrated alpha=0.001 threshold yet.
        history = [make_failed_trial_record(i) for i in range(5)]

        decision = self.evaluator.after_trial(history)

        self.assertTrue(decision.continue_execution)

    def test_six_consecutive_failures_triggers(self) -> None:
        # Hand-verified: binomial_tail_probability(6, 6, 0.3) = 0.000729,
        # below alpha=0.001 -- the first n where 100% failure triggers.
        history = [make_failed_trial_record(i) for i in range(6)]

        decision = self.evaluator.after_trial(history)

        self.assertFalse(decision.continue_execution)
        self.assertEqual(decision.termination_reason, "excessive_failures")
        self.assertIn("6 of 6 trials failed", decision.message)

    def test_all_successful_trials_never_triggers(self) -> None:
        history = [make_trial_record(i) for i in range(140)]

        decision = self.evaluator.after_trial(history)

        self.assertTrue(decision.continue_execution)

    def test_a_healthy_minority_of_failures_does_not_trigger(self) -> None:
        # 10% failure rate, well under the 30% acceptable baseline --
        # should never look statistically excessive.
        history = [make_trial_record(i) for i in range(90)] + [
            make_failed_trial_record(90 + i) for i in range(10)
        ]

        decision = self.evaluator.after_trial(history)

        self.assertTrue(decision.continue_execution)

    def test_reaching_max_trials_still_reports_maximum_trials_first(
        self,
    ) -> None:
        # If both conditions are true at once, the pre-existing
        # max_trials path takes precedence -- it's checked first and the
        # run was going to stop either way.
        evaluator = StopPolicyEvaluator(
            StopPolicy(max_trials=6), make_contract()
        )
        history = [make_failed_trial_record(i) for i in range(6)]

        decision = evaluator.after_trial(history)

        self.assertEqual(decision.termination_reason, "maximum_trials")


if __name__ == "__main__":
    unittest.main()
