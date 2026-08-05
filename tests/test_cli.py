"""Focused tests for CLI exit-code and diagnostic behavior."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from black_box_optimizer.cli import _build_parser, _ProgressReporter, main
from black_box_optimizer.config_loader import ConfigurationError
from black_box_optimizer.models import (
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
)
from black_box_optimizer.records import TrialRecord
from black_box_optimizer.reporting import ReportingError


def make_record(
    trial_id: int,
    execution_status: str = "completed",
    metrics_status: str = "valid",
    metrics: dict | None = None,
    error_message: str | None = None,
) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parameters={},
        metrics={} if metrics is None else metrics,
        execution_status=execution_status,
        metrics_status=metrics_status,
        runtime_seconds=1.0,
        exit_code=0 if execution_status == "completed" else 1,
        timed_out=execution_status == "timed_out",
        error_message=error_message,
    )


def make_session(status: str, termination_reason: str):
    result = SimpleNamespace(
        status=status,
        termination_reason=termination_reason,
        attempted_count=2,
        pareto_count=1,
    )
    configuration = SimpleNamespace(
        stop_policy=SimpleNamespace(max_trials=2),
        optimization=SimpleNamespace(objectives=()),
    )
    return SimpleNamespace(
        configuration=configuration,
        run_directory=SimpleNamespace(path=Path("run-output")),
        run=Mock(return_value=result),
    )


class CliTests(unittest.TestCase):
    def test_entry_points_have_distinct_help_names(self) -> None:
        self.assertEqual(
            _build_parser("python -m black_box_optimizer").prog,
            "python -m black_box_optimizer",
        )
        self.assertEqual(
            _build_parser("hyperloop-optimizer").prog,
            "hyperloop-optimizer",
        )

    def test_result_statuses_map_to_stable_exit_codes(self) -> None:
        cases = (
            ("completed", "maximum_trials", 0),
            ("no_eligible_trials", "search_exhausted", 0),
            ("cancelled", "user_cancelled", 130),
            ("failed", "fatal_error", 1),
        )
        for status, reason, expected in cases:
            with self.subTest(status=status):
                session = make_session(status, reason)
                with patch(
                    "black_box_optimizer.cli.initialize_application",
                    return_value=session,
                ):
                    with redirect_stdout(io.StringIO()):
                        code = main(["config.json"])
                self.assertEqual(code, expected)

    def test_configuration_error_returns_two(self) -> None:
        error = ConfigurationError(("root: invalid",))
        stderr = io.StringIO()
        with patch(
            "black_box_optimizer.cli.initialize_application",
            side_effect=error,
        ):
            with redirect_stderr(stderr):
                code = main(["config.json"])

        self.assertEqual(code, 2)
        self.assertIn("invalid project configuration", stderr.getvalue())

    def test_initialization_interrupt_returns_130(self) -> None:
        stderr = io.StringIO()
        with patch(
            "black_box_optimizer.cli.initialize_application",
            side_effect=KeyboardInterrupt(),
        ):
            with redirect_stderr(stderr):
                code = main(["config.json"])

        self.assertEqual(code, 130)
        self.assertIn("cancelled during initialization", stderr.getvalue())

    def test_os_error_during_initialization_returns_one(self) -> None:
        stderr = io.StringIO()
        with patch(
            "black_box_optimizer.cli.initialize_application",
            side_effect=OSError("disk full"),
        ):
            with redirect_stderr(stderr):
                code = main(["config.json"])

        self.assertEqual(code, 1)
        self.assertIn("Optimization failed", stderr.getvalue())
        self.assertIn("disk full", stderr.getvalue())

    def test_reporting_error_during_run_returns_one(self) -> None:
        session = make_session("completed", "maximum_trials")
        session.run.side_effect = ReportingError("could not write summary")
        stderr = io.StringIO()
        with patch(
            "black_box_optimizer.cli.initialize_application",
            return_value=session,
        ):
            with redirect_stderr(stderr):
                code = main(["config.json"])

        self.assertEqual(code, 1)
        self.assertIn("Optimization failed", stderr.getvalue())
        self.assertIn("could not write summary", stderr.getvalue())

    def test_mid_run_interrupt_reports_completed_count_not_initialization(
        self,
    ) -> None:
        # Regression test: controller.py re-raises KeyboardInterrupt when it
        # lands in an unsafe-to-cancel state (mid-trial), which used to be
        # reported identically to a pre-launch interrupt, even though real
        # trials had already completed and been checkpointed to disk.
        session = make_session("cancelled", "user_cancelled")
        captured_callback = {}

        def fake_initialize_application(
            configuration_path, output_dir, on_trial_complete=None
        ):
            captured_callback["callback"] = on_trial_complete
            return session

        def run_with_one_completed_trial_then_interrupt():
            captured_callback["callback"](make_record(0))
            captured_callback["callback"](make_record(1))
            raise KeyboardInterrupt()

        session.run.side_effect = run_with_one_completed_trial_then_interrupt

        stderr = io.StringIO()
        with patch(
            "black_box_optimizer.cli.initialize_application",
            side_effect=fake_initialize_application,
        ):
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                code = main(["config.json"])

        self.assertEqual(code, 130)
        message = stderr.getvalue()
        self.assertIn("while a trial was in progress", message)
        self.assertIn("2 trial(s) completed", message)
        self.assertNotIn("during initialization", message)


class ProgressReporterTests(unittest.TestCase):
    def make_contract(self) -> OptimizationContract:
        return OptimizationContract(
            parameters=(
                ParameterDefinition("x", ParameterKind.FLOAT, 0.0, 1.0),
            ),
            objectives=(
                Objective("accuracy", Direction.MAXIMIZE),
                Objective("loss", Direction.MINIMIZE),
            ),
        )

    def test_tracks_best_value_per_objective_by_direction(self) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(3, self.make_contract())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            reporter.on_trial_complete(
                make_record(0, metrics={"accuracy": 0.5, "loss": 0.5})
            )
            reporter.on_trial_complete(
                make_record(1, metrics={"accuracy": 0.9, "loss": 0.2})
            )
            reporter.on_trial_complete(
                make_record(2, metrics={"accuracy": 0.3, "loss": 0.8})
            )
            reporter.print_best_summary()

        output = stdout.getvalue()
        self.assertIn("Best accuracy: 0.9000 (trial 1)", output)
        self.assertIn("Best loss: 0.2000 (trial 1)", output)

    def test_a_strict_tie_does_not_overwrite_the_first_best(self) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(2, self.make_contract())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            reporter.on_trial_complete(
                make_record(0, metrics={"accuracy": 0.5, "loss": 0.5})
            )
            reporter.on_trial_complete(
                make_record(1, metrics={"accuracy": 0.5, "loss": 0.5})
            )
            reporter.print_best_summary()

        self.assertIn("trial 0", stdout.getvalue())
        self.assertNotIn("trial 1", stdout.getvalue())

    def test_failure_reason_categorizes_execution_failure(self) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(1, self.make_contract())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            reporter.on_trial_complete(
                make_record(
                    0,
                    execution_status="process_failed",
                    metrics_status="missing",
                    error_message="Worker exited with code 2",
                )
            )
            reporter.print_failure_summary()

        output = stdout.getvalue()
        self.assertIn("process_failed: 1", output)
        self.assertIn("Worker exited with code 2", output)

    def test_failure_reason_categorizes_bad_metrics_on_a_completed_run(
        self,
    ) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(1, self.make_contract())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            reporter.on_trial_complete(
                make_record(0, metrics_status="malformed")
            )
            reporter.print_failure_summary()

        self.assertIn("metrics_malformed: 1", stdout.getvalue())

    def test_failure_reason_categorizes_a_missing_objective_key(
        self,
    ) -> None:
        # execution succeeded, metrics parsed fine, but "loss" was never
        # declared by the worker -- the only remaining way is_eligible()
        # can reject an otherwise-valid-looking record.
        reporter = _ProgressReporter()
        reporter.set_target(1, self.make_contract())

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            reporter.on_trial_complete(
                make_record(0, metrics={"accuracy": 0.9})
            )
            reporter.print_failure_summary()

        self.assertIn("missing_objective_metric: 1", stdout.getvalue())

    def test_print_failure_summary_is_silent_when_nothing_failed(
        self,
    ) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(1, self.make_contract())

        with redirect_stdout(io.StringIO()):
            reporter.on_trial_complete(
                make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})
            )

        # Isolated from the progress bar's own (expected) stdout writes --
        # this only checks print_failure_summary()'s own output.
        failure_summary_output = io.StringIO()
        with redirect_stdout(failure_summary_output):
            reporter.print_failure_summary()

        self.assertEqual(failure_summary_output.getvalue(), "")

    def test_completed_count_tracks_trials_processed_so_far(self) -> None:
        reporter = _ProgressReporter()
        reporter.set_target(3, self.make_contract())
        self.assertEqual(reporter.completed_count, 0)

        with redirect_stdout(io.StringIO()):
            reporter.on_trial_complete(
                make_record(0, metrics={"accuracy": 0.9, "loss": 0.1})
            )
            self.assertEqual(reporter.completed_count, 1)

            reporter.on_trial_complete(
                make_record(1, metrics={"accuracy": 0.9, "loss": 0.1})
            )
            self.assertEqual(reporter.completed_count, 2)


if __name__ == "__main__":
    unittest.main()
