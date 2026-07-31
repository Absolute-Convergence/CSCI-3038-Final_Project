"""Focused tests for TrialHistory."""

from __future__ import annotations

import unittest

from black_box_optimizer.history import TrialHistory
from black_box_optimizer.records import TrialRecord


def make_trial_record(trial_id: int) -> TrialRecord:
    """Build a minimal, valid TrialRecord for testing history behavior."""
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


class TrialHistoryTests(unittest.TestCase):
    """Verify append, snapshot, and ordering guarantees."""

    # Tests that verify the normal append and snapshot behavior

    def test_new_history_is_empty(self) -> None:
        history = TrialHistory()
        self.assertEqual(history.snapshot(), ())
        self.assertEqual(len(history.snapshot()), 0)

    def test_append_adds_to_snapshot(self) -> None:
        history = TrialHistory()
        history.append(make_trial_record(0))

        snapshot = history.snapshot()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].trial_id, 0)

    def test_multiple_appends_preserve_order(self) -> None:
        history = TrialHistory()
        for trial_id in (0, 1, 2):
            history.append(make_trial_record(trial_id))

        snapshot = history.snapshot()
        self.assertEqual([r.trial_id for r in snapshot], [0, 1, 2])
        self.assertEqual(len(snapshot), 3)

    def test_snapshot_returns_a_tuple(self) -> None:
        history = TrialHistory()
        history.append(make_trial_record(0))
        self.assertIsInstance(history.snapshot(), tuple)

    def test_earlier_snapshot_is_unaffected_by_later_appends(self) -> None:
        history = TrialHistory()
        history.append(make_trial_record(0))
        first_snapshot = history.snapshot()

        history.append(make_trial_record(1))

        self.assertEqual(len(first_snapshot), 1)
        self.assertEqual(len(history.snapshot()), 2)

    # Tests that reject duplicate or out-of-order trial ids

    def test_duplicate_trial_id_rejected(self) -> None:
        history = TrialHistory()
        history.append(make_trial_record(0))

        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            history.append(make_trial_record(0))

    def test_out_of_order_trial_id_rejected(self) -> None:
        history = TrialHistory()
        history.append(make_trial_record(5))

        with self.assertRaisesRegex(ValueError, "duplicate or out of order"):
            history.append(make_trial_record(3))


if __name__ == "__main__":
    unittest.main()
