"""Focused tests for StopDecision and StopPolicyEvaluator."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from black_box_optimizer.models import StopPolicy
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.stop_policy import StopDecision, StopPolicyEvaluator


def make_trial_record(trial_id: int) -> TrialRecord:
    """Build a minimal, valid TrialRecord for filling out fake history."""
    return TrialRecord(
        trial_id=trial_id,
        parameters={"learning_rate": 0.01},
        metrics={"accuracy": 0.9},
        execution_status="completed",
        metrics_status="valid",
        runtime_seconds=1.0,
        exit_code=0,
        timed_out=False,
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


if __name__ == "__main__":
    unittest.main()
