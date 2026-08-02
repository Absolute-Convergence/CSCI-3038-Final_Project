"""
Focused tests for RunDirectory and create_run_directory(), following
TDS section 9.

These tests cover the persistence pieces that already exist today,
including run creation, per-trial paths, and atomic history.csv
checkpoints.

Keep an eye out (!!!) for notes about zero-indexed trial directories,
metric column unions, checkpoint replacement, and failure cleanup.

COVERAGE NOTE!!!! These tests are stable, not temporary. The missing
finalization files, resolved config output, and stdout/stderr
preservation will be new features when they arrive, so they should get
new test classes instead of requiring these tests to be rewritten.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.persistence import (
    CheckpointError,
    RunDirectory,
    create_run_directory,
)
from black_box_optimizer.records import TrialRecord


def make_contract() -> OptimizationContract:
    """Build the contract used throughout the persistence tests."""
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.01, 0.1
            ),
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16)
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


def make_record(
    trial_id: int,
    metrics: dict[str, float] | None = None,
    execution_status: str = "completed",
    metrics_status: str = "valid",
    error_message: str | None = None,
) -> TrialRecord:
    """Build a small TrialRecord with sensible defaults for these tests."""
    return TrialRecord(
        trial_id=trial_id,
        parameters={"learning_rate": 0.05, "batch_size": 8},
        metrics={} if metrics is None else metrics,
        execution_status=execution_status,
        metrics_status=metrics_status,
        runtime_seconds=1.5,
        exit_code=0 if execution_status == "completed" else 1,
        timed_out=False,
        error_message=error_message,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read history.csv back into plain dictionaries for easier checks."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class CreateRunDirectoryTests(unittest.TestCase):
    """Tests for creating fresh run directories without overwriting old ones."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.base = Path(self._tmpdir.name)

    def test_creates_a_new_directory_under_base(self) -> None:
        run_directory = create_run_directory(self.base)

        self.assertTrue(run_directory.path.is_dir())
        self.assertEqual(run_directory.path.parent, self.base)

    def test_creates_a_trials_subdirectory(self) -> None:
        run_directory = create_run_directory(self.base)

        self.assertTrue((run_directory.path / "trials").is_dir())

    def test_each_call_creates_a_distinct_directory(self) -> None:
        first = create_run_directory(self.base)
        second = create_run_directory(self.base)

        self.assertNotEqual(first.path, second.path)

    def test_never_resumes_or_overwrites_an_existing_run(self) -> None:
        # Pin both parts of the generated name so exist_ok False gets
        # tested for real instead of trusting collisions never happen
        fixed_now = datetime(2026, 1, 1, 12, 0, 0)

        with (
            patch(
                "black_box_optimizer.persistence.secrets.token_hex",
                return_value="fixed",
            ),
            patch(
                "black_box_optimizer.persistence.datetime"
            ) as mock_datetime,
        ):
            mock_datetime.now.return_value = fixed_now
            create_run_directory(self.base)

            with self.assertRaises(FileExistsError):
                create_run_directory(self.base)


class TrialPathTests(unittest.TestCase):
    """Tests for one trial's directory and metrics.csv path."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = RunDirectory(Path(self._tmpdir.name) / "run")

    def test_trial_directory_is_created_and_zero_padded(self) -> None:
        directory = self.run_directory.trial_directory(3)

        self.assertTrue(directory.is_dir())
        self.assertEqual(directory.name, "trial_0003")

    def test_metrics_path_lives_inside_its_trial_directory(self) -> None:
        path = self.run_directory.metrics_path(7)

        self.assertEqual(path.name, "metrics.csv")
        self.assertEqual(path.parent.name, "trial_0007")
        self.assertTrue(path.parent.is_dir())

        # metrics_path creates the directory but not the file itself
        self.assertFalse(path.exists())


class CheckpointTests(unittest.TestCase):
    """Tests for flattened and atomically replaced history.csv checkpoints."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = RunDirectory(Path(self._tmpdir.name) / "run")
        self.contract = make_contract()

    def test_empty_history_writes_header_only(self) -> None:
        self.run_directory.checkpoint([], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(rows, [])

    def test_writes_fixed_columns_correctly(self) -> None:
        record = make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})

        self.run_directory.checkpoint([record], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["trial_id"], "0")
        self.assertEqual(row["execution_status"], "completed")
        self.assertEqual(row["metrics_status"], "valid")
        self.assertEqual(row["execution_succeeded"], "True")
        self.assertEqual(row["objective_eligible"], "True")
        self.assertEqual(row["runtime_seconds"], "1.5")
        self.assertEqual(row["exit_code"], "0")
        self.assertEqual(row["timed_out"], "False")

    def test_writes_prefixed_param_and_metric_columns(self) -> None:
        record = make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})

        self.run_directory.checkpoint([record], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        row = rows[0]
        self.assertEqual(row["param.learning_rate"], "0.05")
        self.assertEqual(row["param.batch_size"], "8")
        self.assertEqual(row["metric.accuracy"], "0.9")
        self.assertEqual(row["metric.loss"], "0.1")

    def test_objective_eligible_reflects_real_is_eligible(self) -> None:
        ineligible = make_record(
            0,
            metrics_status="missing",
            metrics={},
            execution_status="process_failed",
        )

        self.run_directory.checkpoint([ineligible], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(rows[0]["objective_eligible"], "False")

    def test_metric_columns_are_the_union_across_all_records(self) -> None:
        # The first trial has no loss metric but the second one does
        # history csv still needs one shared set of columns for both
        first = make_record(0, metrics={"accuracy": 0.9})
        second = make_record(1, metrics={"accuracy": 0.8, "loss": 0.2})

        self.run_directory.checkpoint([first, second], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(rows[0]["metric.accuracy"], "0.9")
        self.assertEqual(rows[0]["metric.loss"], "")
        self.assertEqual(rows[1]["metric.loss"], "0.2")

    def test_error_message_none_is_an_empty_cell_not_the_string_none(self) -> None:
        record = make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})

        self.run_directory.checkpoint([record], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(rows[0]["error_message"], "")

    def test_exit_code_none_is_an_empty_cell_not_the_string_none(self) -> None:
        # exit_code is a different field and type from error_message
        # so it gets its own test instead of assuming one covers both
        record = TrialRecord(
            trial_id=0,
            parameters={"learning_rate": 0.05, "batch_size": 8},
            metrics={},
            execution_status="timed_out",
            metrics_status="missing",
            runtime_seconds=30.0,
            exit_code=None,
            timed_out=True,
        )

        self.run_directory.checkpoint([record], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(rows[0]["exit_code"], "")

    def test_error_message_is_written_when_present(self) -> None:
        record = make_record(
            0,
            execution_status="process_failed",
            metrics_status="missing",
            error_message="worker exited with code 1",
        )

        self.run_directory.checkpoint([record], self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(
            rows[0]["error_message"],
            "worker exited with code 1",
        )

    def test_rows_are_written_in_trial_id_order(self) -> None:
        records = [
            make_record(i, metrics={"accuracy": 0.5, "loss": 0.5})
            for i in range(3)
        ]

        self.run_directory.checkpoint(records, self.contract)
        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(
            [row["trial_id"] for row in rows],
            ["0", "1", "2"],
        )

    def test_second_checkpoint_replaces_the_first(self) -> None:
        first = make_record(0, metrics={"accuracy": 0.5, "loss": 0.5})
        self.run_directory.checkpoint([first], self.contract)

        second = make_record(1, metrics={"accuracy": 0.6, "loss": 0.4})
        self.run_directory.checkpoint([first, second], self.contract)

        rows = read_csv_rows(self.run_directory.path / "history.csv")

        self.assertEqual(len(rows), 2)

    def test_no_leftover_temp_files_after_checkpoint(self) -> None:
        record = make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})

        self.run_directory.checkpoint([record], self.contract)
        leftovers = list(self.run_directory.path.glob("*.tmp"))

        self.assertEqual(leftovers, [])

    def test_failure_retains_the_previously_committed_file(self) -> None:
        first = make_record(0, metrics={"accuracy": 0.5, "loss": 0.5})
        self.run_directory.checkpoint([first], self.contract)

        before = (self.run_directory.path / "history.csv").read_text()
        second = make_record(1, metrics={"accuracy": 0.6, "loss": 0.4})

        with patch(
            "black_box_optimizer.persistence.os.replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(CheckpointError):
                self.run_directory.checkpoint(
                    [first, second],
                    self.contract,
                )

        after = (self.run_directory.path / "history.csv").read_text()

        self.assertEqual(before, after)

    def test_failure_does_not_leave_a_temp_file_behind(self) -> None:
        record = make_record(0, metrics={"accuracy": 0.5, "loss": 0.5})

        with patch(
            "black_box_optimizer.persistence.os.replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(CheckpointError):
                self.run_directory.checkpoint([record], self.contract)

        leftovers = list(self.run_directory.path.glob("*.tmp"))

        self.assertEqual(leftovers, [])

    def test_failure_message_names_the_most_recent_trial_id(self) -> None:
        # TDS section 10 4 wants the assigned trial identifier included
        # in the error message when one exists
        record = make_record(5, metrics={"accuracy": 0.5, "loss": 0.5})

        with patch(
            "black_box_optimizer.persistence.os.replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(CheckpointError, "trial_id: 5"):
                self.run_directory.checkpoint([record], self.contract)

    def test_failure_message_handles_empty_history_gracefully(self) -> None:
        # Empty history means there is no trial identifier to report
        # this still needs to fail cleanly instead of causing another error
        with patch(
            "black_box_optimizer.persistence.os.replace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaises(CheckpointError):
                self.run_directory.checkpoint([], self.contract)


if __name__ == "__main__":
    unittest.main()