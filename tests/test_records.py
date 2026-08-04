"""Focused tests for TrialRecord and the build_trial_record factory."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from black_box_optimizer.models import CandidateConfiguration
from black_box_optimizer.records import (
    TrialRecord,
    build_internal_error_record,
    build_trial_record,
)


class TrialRecordTests(unittest.TestCase):
    """Verify construction, validation, and immutability of TrialRecord."""

    def make_record(self, **overrides) -> TrialRecord:
        """Build a TrialRecord with sensible defaults for a single test."""
        fields = {
            "trial_id": 1,
            "parameters": {"learning_rate": 0.01},
            "metrics": {"accuracy": 0.9},
            "execution_status": "completed",
            "metrics_status": "valid",
            "runtime_seconds": 1.5,
            "exit_code": 0,
            "timed_out": False,
        }
        fields.update(overrides)
        return TrialRecord(**fields)

    # Tests that check how execution_succeeded reports success.

    def test_execution_succeeded_true_when_completed(self) -> None:
        record = self.make_record(execution_status="completed")
        self.assertTrue(record.execution_succeeded)

    def test_execution_succeeded_false_when_not_completed(self) -> None:
        record = self.make_record(execution_status="timed_out")
        self.assertFalse(record.execution_succeeded)

    def test_internal_error_is_a_valid_unsuccessful_status(self) -> None:
        record = self.make_record(execution_status="internal_error")
        self.assertFalse(record.execution_succeeded)

    # Tests that confirm a TrialRecord truly cannot be changed once built.

    def test_parameters_and_metrics_are_copied_and_read_only(self) -> None:
        parameters = {"learning_rate": 0.01}
        metrics = {"accuracy": 0.9}
        record = self.make_record(parameters=parameters, metrics=metrics)
        parameters["learning_rate"] = 0.05
        metrics["accuracy"] = 0.0

        self.assertEqual(record.parameters["learning_rate"], 0.01)
        self.assertEqual(record.metrics["accuracy"], 0.9)
        with self.assertRaises(TypeError):
            record.parameters["learning_rate"] = 0.2  # type: ignore[index]
        with self.assertRaises(TypeError):
            record.metrics["accuracy"] = 0.2  # type: ignore[index]

    def test_record_is_frozen(self) -> None:
        record = self.make_record()
        with self.assertRaises(FrozenInstanceError):
            record.trial_id = 2  # type: ignore[misc]

    # Tests that make sure bad field values get rejected.

    def test_negative_trial_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            self.make_record(trial_id=-1)

    def test_negative_runtime_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            self.make_record(runtime_seconds=-0.1)

    def test_invalid_execution_status_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution_status"):
            self.make_record(execution_status="banana")

    def test_invalid_metrics_status_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "metrics_status"):
            self.make_record(metrics_status="banana")

    def test_boolean_trial_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            self.make_record(trial_id=True)

    def test_non_numeric_runtime_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.make_record(runtime_seconds="fast")

    def test_infinite_runtime_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            self.make_record(runtime_seconds=float("inf"))

    def test_non_integer_exit_code_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            self.make_record(exit_code="0")

    def test_boolean_exit_code_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            self.make_record(exit_code=True)

    def test_non_boolean_timed_out_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "bool"):
            self.make_record(timed_out="yes")

    def test_non_string_error_message_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "string"):
            self.make_record(error_message=404)


class BuildTrialRecordTests(unittest.TestCase):
    """Verify build_trial_record against each metrics outcome."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        self.candidate = CandidateConfiguration(
            parameters={"learning_rate": 0.01, "batch_size": 16}
        )

    def write_csv(self, name: str, text: str) -> Path:
        """Create a temporary metrics CSV file for an individual test case."""
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def make_execution_result(self, **overrides) -> dict:
        """Build a fake runner.execute() result with sensible defaults."""
        fields = {
            "execution_status": "completed",
            "runtime_seconds": 2.0,
            "exit_code": 0,
            "timed_out": False,
            "error_message": None,
        }
        fields.update(overrides)
        return fields

    # Tests that check each of the four possible metrics_status outcomes.

    def test_valid_metrics_produce_valid_status(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy,loss\n0.9,0.1\n")
        record = build_trial_record(
            self.candidate, 3, path, self.make_execution_result()
        )

        self.assertEqual(record.metrics_status, "valid")
        self.assertEqual(record.metrics["accuracy"], 0.9)
        self.assertEqual(record.parameters, self.candidate.parameters)
        self.assertEqual(record.trial_id, 3)

    def test_missing_metrics_file_produces_missing_status(self) -> None:
        missing = self.directory / "missing.csv"
        record = build_trial_record(
            self.candidate, 3, missing, self.make_execution_result()
        )

        self.assertEqual(record.metrics_status, "missing")
        self.assertEqual(record.metrics, {})

    def test_malformed_csv_produces_malformed_status(self) -> None:
        path = self.write_csv("metrics.csv", "")
        record = build_trial_record(
            self.candidate, 3, path, self.make_execution_result()
        )

        self.assertEqual(record.metrics_status, "malformed")
        self.assertEqual(record.metrics, {})

    def test_nonfinite_value_produces_nonfinite_status(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\nnan\n")
        record = build_trial_record(
            self.candidate, 3, path, self.make_execution_result()
        )

        self.assertEqual(record.metrics_status, "nonfinite")
        self.assertEqual(record.metrics, {})

    # A test that confirms execution details pass through from the runner
    # untouched, since build_trial_record must not invent or alter them.

    def test_execution_fields_pass_through_from_result(self) -> None:
        path = self.write_csv("metrics.csv", "accuracy\n1\n")
        execution_result = self.make_execution_result(
            execution_status="timed_out",
            runtime_seconds=120.0,
            exit_code=None,
            timed_out=True,
            error_message="Worker exceeded 120.0-second timeout",
        )
        record = build_trial_record(self.candidate, 3, path, execution_result)

        self.assertEqual(record.execution_status, "timed_out")
        self.assertEqual(record.runtime_seconds, 120.0)
        self.assertIsNone(record.exit_code)
        self.assertTrue(record.timed_out)
        self.assertEqual(
            record.error_message, "Worker exceeded 120.0-second timeout"
        )

    def test_internal_error_factory_preserves_authorized_attempt(self) -> None:
        record = build_internal_error_record(
            self.candidate,
            4,
            "executing",
            RuntimeError("runner contract broke"),
            0.25,
        )

        self.assertEqual(record.trial_id, 4)
        self.assertEqual(record.parameters, self.candidate.parameters)
        self.assertEqual(record.execution_status, "internal_error")
        self.assertEqual(record.metrics_status, "missing")
        self.assertEqual(record.metrics, {})
        self.assertEqual(record.runtime_seconds, 0.25)
        self.assertIsNone(record.exit_code)
        self.assertFalse(record.timed_out)
        self.assertIn(
            "RuntimeError: runner contract broke",
            record.error_message,
        )

    def test_internal_error_diagnostic_is_bounded(self) -> None:
        record = build_internal_error_record(
            self.candidate,
            4,
            "recording",
            RuntimeError("x" * 2_000),
            0.0,
        )

        self.assertEqual(len(record.error_message), 1_000)


if __name__ == "__main__":
    unittest.main()
