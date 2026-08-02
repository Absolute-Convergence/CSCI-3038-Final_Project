"""Focused tests for initialization outside the lifecycle controller."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
