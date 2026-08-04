"""Focused tests for initialization outside the lifecycle controller."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from black_box_optimizer.application import initialize_application
from black_box_optimizer.controller import ControllerState


_WORKER = Path(__file__).parent / "fixtures" / "cli_worker.py"


def write_configuration(path: Path, max_trials: int = 1) -> None:
    document = {
        "worker": {
            "command": [sys.executable, str(_WORKER)],
            "metrics_argument": "--metrics-out",
            "timeout_seconds": 10.0,
        },
        "optimization": {
            "parameters": [
                {
                    "name": "x",
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": max_trials,
                }
            ],
            "objectives": [
                {"metric_name": "score", "direction": "maximize"},
                {"metric_name": "cost", "direction": "minimize"},
            ],
        },
        "algorithm": {"name": "random_search", "seed": 4},
        "stop_policy": {"max_trials": max_trials},
    }
    path.write_text(json.dumps(document), encoding="utf-8")


class ApplicationInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.configuration_path = self.directory / "configuration.json"
        write_configuration(self.configuration_path)

    def test_initialization_writes_config_before_controller_runs(self) -> None:
        session = initialize_application(
            self.configuration_path,
            self.directory / "runs",
        )

        self.assertEqual(session.controller.state, ControllerState.SELECTING)
        self.assertEqual(session.controller.history, ())
        self.assertTrue(
            (session.run_directory.path / "resolved_config.json").is_file()
        )
        self.assertFalse(
            (session.run_directory.path / "history.csv").exists()
        )

    def test_initialized_session_runs_to_authoritative_outputs(self) -> None:
        session = initialize_application(
            self.configuration_path,
            self.directory / "runs",
        )

        result = session.run()

        self.assertEqual(result.status, "completed")
        self.assertEqual(session.controller.state, ControllerState.STOPPED)
        self.assertEqual(result.attempted_count, 1)
        for filename in (
            "history.csv",
            "pareto_front.csv",
            "pareto_front.png",
            "resolved_config.json",
            "summary.txt",
        ):
            self.assertTrue((session.run_directory.path / filename).is_file())

    def test_repeated_runs_do_not_reuse_an_older_pareto_front(self) -> None:
        output_directory = self.directory / "runs"
        write_configuration(self.configuration_path, max_trials=2)
        first_session = initialize_application(
            self.configuration_path,
            output_directory,
        )
        first_result = first_session.run()
        first_pareto_path = (
            first_session.run_directory.path / "pareto_front.csv"
        )
        first_pareto_before = first_pareto_path.read_bytes()

        write_configuration(self.configuration_path, max_trials=1)
        second_session = initialize_application(
            self.configuration_path,
            output_directory,
        )
        second_result = second_session.run()

        self.assertNotEqual(
            first_session.run_directory.path,
            second_session.run_directory.path,
        )
        self.assertEqual(first_result.attempted_count, 2)
        self.assertEqual(first_result.pareto_count, 2)
        self.assertEqual(second_result.attempted_count, 1)
        self.assertEqual(second_result.pareto_count, 1)
        self.assertEqual(len(tuple(output_directory.iterdir())), 2)
        self.assertEqual(first_pareto_path.read_bytes(), first_pareto_before)

        second_pareto_path = (
            second_session.run_directory.path / "pareto_front.csv"
        )
        with second_pareto_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["trial_id"] for row in rows], ["0"])


if __name__ == "__main__":
    unittest.main()
