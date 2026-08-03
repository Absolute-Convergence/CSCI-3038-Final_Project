"""Real application-composition pipeline test, using NSGA2 instead of
RandomSearch.

test_application_real_worker.py already proves the Application/Controller/
runner/worker/metrics/history/reporting stack composes correctly end to
end with RandomSearch. This file proves the same composition holds for
NSGA2 specifically -- nothing here is mocked or simulated, every trial is
a real `python worker.py` subprocess training a real (tiny) model.

The Iris worker's CLI requires all four of learning_rate, hidden_size,
epochs, and batch_size, so the contract needs all four ParameterDefinition
entries. _default_population_size() counts every declared parameter
regardless of range width, so this contract's population size is fixed at
clamp(2 * 4, 4, 10) = 8. max_trials is set to 9 on purpose: 8 real trials
to complete generation 0 (random), plus 1 real trial bred from it -- so
this test actually crosses a generation boundary through the real
pipeline, not just generation 0.
"""

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

_GENERATION_SIZE = 8
_MAX_TRIALS = _GENERATION_SIZE + 1


class RealApplicationNSGA2PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)

    def test_real_iris_run_crosses_a_generation_boundary_with_nsga2(
        self,
    ) -> None:
        configuration_path = self.directory / "iris_nsga2_test_config.json"
        configuration_path.write_text(
            json.dumps(_iris_nsga2_test_document()),
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
        self.assertEqual(result.attempted_count, _MAX_TRIALS)
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

        # The 9th trial is generation 1's first real, bred candidate --
        # this is the actual generation-boundary crossing this test exists
        # to prove, executed through the real subprocess pipeline like
        # every other trial, not synthesized.
        bred_trial = result.history[_GENERATION_SIZE]
        self.assertEqual(bred_trial.trial_id, _GENERATION_SIZE)
        self.assertEqual(bred_trial.execution_status, "completed")

        # Sanity check that NSGA2 actually produced varied candidates
        # rather than proposing the same configuration nine times over.
        distinct_configurations = {
            tuple(sorted(record.parameters.items()))
            for record in result.history
        }
        self.assertGreater(len(distinct_configurations), 1)

        for filename in (
            "history.csv",
            "pareto_front.csv",
            "pareto_front.png",
            "resolved_config.json",
            "summary.txt",
        ):
            self.assertTrue((session.run_directory.path / filename).is_file())


def _iris_nsga2_test_document() -> dict[str, object]:
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
                    "maximum": 0.1,
                },
                {
                    "name": "hidden_size",
                    "kind": "integer",
                    "minimum": 4,
                    "maximum": 16,
                },
                {
                    "name": "epochs",
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 2,
                },
                {
                    "name": "batch_size",
                    "kind": "categorical",
                    "choices": [8, 16, 32],
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
            ],
        },
        "algorithm": {"name": "nsga2", "seed": 42},
        "stop_policy": {"max_trials": _MAX_TRIALS},
    }


if __name__ == "__main__":
    unittest.main()
