"""Real application-composition integration through the Iris worker."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from black_box_optimizer.application import initialize_application
from black_box_optimizer.controller import ControllerState


_REPOSITORY = Path(__file__).resolve().parents[2]
_IRIS_WORKER = _REPOSITORY / "examples" / "iris_torch" / "worker.py"


class RealApplicationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def test_real_iris_run_reaches_stopped_with_complete_outputs(self) -> None:
        configuration_path = self.directory / "iris_test_config.json"
        configuration_path.write_text(
            json.dumps(_iris_test_document()),
            encoding="utf-8",
        )
        session = initialize_application(
            configuration_path,
            self.directory / "runs",
        )

        result = session.run()

        self.assertEqual(session.controller.state, ControllerState.STOPPED)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "maximum_trials")
        self.assertEqual(result.attempted_count, 2)
        self.assertGreaterEqual(result.pareto_count, 1)
        for expected_id, record in enumerate(result.history):
            self.assertEqual(record.trial_id, expected_id)
            self.assertEqual(record.execution_status, "completed")
            self.assertEqual(record.metrics_status, "valid")
            self.assertTrue(
                session.run_directory.trial_directory(expected_id)
                .joinpath("stdout.txt")
                .is_file()
            )
            self.assertTrue(
                session.run_directory.trial_directory(expected_id)
                .joinpath("stderr.txt")
                .is_file()
            )

        for filename in (
            "history.csv",
            "pareto_front.csv",
            "pareto_front.png",
            "resolved_config.json",
            "summary.txt",
        ):
            self.assertTrue((session.run_directory.path / filename).is_file())


def _iris_test_document() -> dict[str, object]:
    return {
        "worker": {
            "command": [sys.executable, str(_IRIS_WORKER)],
            "metrics_argument": "--metrics-out",
            "timeout_seconds": 30.0,
        },
        "optimization": {
            "parameters": [
                {
                    "name": "learning_rate",
                    "kind": "float",
                    "minimum": 0.01,
                    "maximum": 0.0101,
                },
                {
                    "name": "hidden_size",
                    "kind": "integer",
                    "minimum": 4,
                    "maximum": 4,
                },
                {
                    "name": "epochs",
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 1,
                },
                {
                    "name": "batch_size",
                    "kind": "categorical",
                    "choices": [8],
                },
            ],
            "objectives": [
                {
                    "metric_name": "validation_accuracy",
                    "direction": "maximize",
                },
                {
                    "metric_name": "validation_loss",
                    "direction": "minimize",
                },
                {
                    "metric_name": "training_time_seconds",
                    "direction": "minimize",
                },
            ],
        },
        "algorithm": {"name": "random_search", "seed": 42},
        "stop_policy": {"max_trials": 2},
    }


if __name__ == "__main__":
    unittest.main()
