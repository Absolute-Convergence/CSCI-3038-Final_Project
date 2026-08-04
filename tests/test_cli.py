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


if __name__ == "__main__":
    unittest.main()
