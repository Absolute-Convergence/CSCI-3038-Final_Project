"""
The full pipeline slice, using the real merged runner.py and the real
Iris worker.py as an actual subprocess -- nothing simulated.

test_one_trial_slice.py deliberately writes a fake metrics CSV to stand
in for "the worker already ran," since runner.py wasn't merged yet when it
was written. Now that runner.py and worker.py both exist for real, this
file chains everything all the way through: RandomSearch proposes a
candidate, runner.execute() launches worker.py as a real subprocess,
worker.py actually trains a model on the bundled Iris data and writes a
metrics file, build_trial_record() reads it back through the real
metrics.py, the record is appended to a real TrialHistory, and
StopPolicyEvaluator gates each step. No project code is mocked.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

from black_box_optimizer import runner
from black_box_optimizer.history import TrialHistory
from black_box_optimizer.models import (
    AlgorithmSpec,
    CandidateConfiguration,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    StopPolicy,
    WorkerSpec,
)
from black_box_optimizer.records import build_trial_record
from black_box_optimizer.search.registry import create_algorithm
from black_box_optimizer.stop_policy import StopPolicyEvaluator

_WORKER_PATH = (
    Path(__file__).parent.parent.parent
    / "examples" / "iris_torch" / "worker.py"
)


def make_contract() -> OptimizationContract:
    """Mirrors the exact hyperparameters worker.py's CLI accepts, kept
    small so every real subprocess trial trains fast."""
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.01, 0.1
            ),
            ParameterDefinition(
                "hidden_size", ParameterKind.INTEGER, 4, 16
            ),
            ParameterDefinition(
                "epochs", ParameterKind.INTEGER, 1, 3
            ),
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16, 32)
            ),
        ),
        objectives=(
            Objective("validation_accuracy", Direction.MAXIMIZE),
            Objective("validation_loss", Direction.MINIMIZE),
        ),
    )


def make_worker_spec() -> WorkerSpec:
    """Points runner.py at the real Iris worker script."""
    return WorkerSpec(
        command=(sys.executable, str(_WORKER_PATH)),
        metrics_argument="--metrics-out",
        timeout_seconds=60.0,
    )


class FullPipelineRealWorkerTests(unittest.TestCase):
    """Run real trials through search, runner, worker, metrics, records,
    history, and stop_policy together, with nothing mocked."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)

    def test_multi_trial_run_with_real_subprocess_worker(self) -> None:
        contract = make_contract()
        worker_spec = make_worker_spec()
        history = TrialHistory()
        stop_eval = StopPolicyEvaluator(StopPolicy(max_trials=3))
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=7)
        )

        for trial_id in range(3):
            gate = stop_eval.before_trial(history.snapshot())
            self.assertTrue(gate.continue_execution)

            proposal = algorithm.propose(contract, history.snapshot())
            self.assertEqual(proposal.status, "candidate")

            metrics_path = self.directory / f"trial_{trial_id}.csv"
            execution_result = runner.execute(
                worker_spec, proposal.candidate, metrics_path
            )
            record = build_trial_record(
                proposal.candidate, trial_id, metrics_path, execution_result
            )

            self.assertEqual(record.execution_status, "completed")
            self.assertEqual(record.metrics_status, "valid")
            self.assertEqual(record.exit_code, 0)
            self.assertFalse(record.timed_out)

            accuracy = record.metrics["validation_accuracy"]
            loss = record.metrics["validation_loss"]
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)
            self.assertTrue(math.isfinite(accuracy))
            self.assertTrue(math.isfinite(loss))
            training_time = record.metrics["training_time_seconds"]
            self.assertTrue(math.isfinite(training_time))

            self.assertEqual(record.parameters, proposal.candidate.parameters)

            history.append(record)

        self.assertEqual(len(history.snapshot()), 3)

        final_gate = stop_eval.after_trial(history.snapshot())
        self.assertFalse(final_gate.continue_execution)
        self.assertEqual(final_gate.termination_reason, "maximum_trials")

    def test_launch_failure_flows_through_without_crashing(self) -> None:
        # Points runner.py at an executable that does not exist at all
        # (not just a missing script), the only way to actually trigger
        # a FileNotFoundError instead of the interpreter itself exiting
        # non-zero.
        contract = make_contract()
        broken_worker_spec = WorkerSpec(
            command=(str(self.directory / "no_such_executable"),),
            metrics_argument="--metrics-out",
            timeout_seconds=10.0,
        )
        history = TrialHistory()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=3)
        )

        proposal = algorithm.propose(contract, history.snapshot())
        metrics_path = self.directory / "trial_0.csv"
        execution_result = runner.execute(
            broken_worker_spec, proposal.candidate, metrics_path
        )
        record = build_trial_record(
            proposal.candidate, 0, metrics_path, execution_result
        )

        self.assertEqual(record.execution_status, "launch_failed")
        self.assertEqual(record.metrics_status, "missing")
        self.assertIsNotNone(record.error_message)

        # A failed launch is still real evidence and must be recorded.
        history.append(record)
        self.assertEqual(len(history.snapshot()), 1)

    def test_real_subprocess_timeout_flows_through_as_timed_out(self) -> None:
        # A timeout small enough that even Python's own startup can't
        # finish inside it, so this reliably forces a genuine
        # subprocess.TimeoutExpired instead of just running slowly.
        contract = make_contract()
        impatient_worker_spec = WorkerSpec(
            command=(sys.executable, str(_WORKER_PATH)),
            metrics_argument="--metrics-out",
            timeout_seconds=0.01,
        )
        history = TrialHistory()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=11)
        )

        proposal = algorithm.propose(contract, history.snapshot())
        metrics_path = self.directory / "trial_0.csv"
        execution_result = runner.execute(
            impatient_worker_spec, proposal.candidate, metrics_path
        )
        record = build_trial_record(
            proposal.candidate, 0, metrics_path, execution_result
        )

        self.assertEqual(record.execution_status, "timed_out")
        self.assertTrue(record.timed_out)
        self.assertIsNone(record.exit_code)
        self.assertIsNotNone(record.error_message)
        # The worker was killed before it ever reached write_metrics().
        self.assertEqual(record.metrics_status, "missing")
        self.assertFalse(metrics_path.exists())

        history.append(record)
        self.assertEqual(len(history.snapshot()), 1)

    def test_real_worker_crash_flows_through_as_process_failed(self) -> None:
        # batch_size=0 is outside the contract's declared choices, but
        # CandidateConfiguration itself does not enforce a contract's
        # domain -- only the search algorithm avoids values like this.
        # Building the candidate directly here forces worker.py to hit a
        # real, unhandled torch.utils.data.DataLoader ValueError at
        # runtime, the only reliable way to make the real subprocess
        # exit non-zero without touching worker.py's own code.
        worker_spec = make_worker_spec()
        history = TrialHistory()
        candidate = CandidateConfiguration(parameters={
            "learning_rate": 0.05,
            "hidden_size": 8,
            "epochs": 1,
            "batch_size": 0,
        })

        metrics_path = self.directory / "trial_0.csv"
        execution_result = runner.execute(
            worker_spec, candidate, metrics_path
        )
        record = build_trial_record(
            candidate, 0, metrics_path, execution_result
        )

        self.assertEqual(record.execution_status, "process_failed")
        self.assertNotEqual(record.exit_code, 0)
        self.assertFalse(record.timed_out)
        self.assertIsNotNone(record.error_message)
        self.assertLessEqual(len(record.error_message), 1_000)
        self.assertIn("batch_size", record.error_message)
        self.assertIn("Traceback", execution_result["stderr"])
        self.assertEqual(record.metrics_status, "missing")
        self.assertFalse(metrics_path.exists())

        history.append(record)
        self.assertEqual(len(history.snapshot()), 1)

    def test_boundary_parameter_values_survive_real_cli_round_trip(
        self,
    ) -> None:
        # The happy-path test only exercises randomly sampled values.
        # This confirms the exact minimum and maximum of every declared
        # parameter domain still parses correctly through the real
        # argparse CLI and produces a valid trial, not just the middle
        # of each range.
        worker_spec = make_worker_spec()
        boundary_candidates = (
            {
                "learning_rate": 0.01,
                "hidden_size": 4,
                "epochs": 1,
                "batch_size": 8,
            },
            {
                "learning_rate": 0.1,
                "hidden_size": 16,
                "epochs": 3,
                "batch_size": 32,
            },
        )

        for trial_id, parameters in enumerate(boundary_candidates):
            with self.subTest(parameters=parameters):
                history = TrialHistory()
                candidate = CandidateConfiguration(parameters=parameters)
                metrics_path = self.directory / f"boundary_{trial_id}.csv"

                execution_result = runner.execute(
                    worker_spec, candidate, metrics_path
                )
                record = build_trial_record(
                    candidate, trial_id, metrics_path, execution_result
                )

                self.assertEqual(record.execution_status, "completed")
                self.assertEqual(record.metrics_status, "valid")
                self.assertEqual(record.exit_code, 0)

                accuracy = record.metrics["validation_accuracy"]
                loss = record.metrics["validation_loss"]
                self.assertGreaterEqual(accuracy, 0.0)
                self.assertLessEqual(accuracy, 1.0)
                self.assertTrue(math.isfinite(accuracy))
                self.assertTrue(math.isfinite(loss))

                history.append(record)


if __name__ == "__main__":
    unittest.main()
