"""
The one-trial candidate-to-record vertical slice, per AGENTS.md's
Verification Expectations: "Add integration tests across declared
boundaries, beginning with the one-trial candidate-to-record vertical
slice."

Unlike the other test files, this one does not test any single module in
isolation. It chains the real modules together the way the eventual
controller will: RandomSearch proposes a candidate, a metrics CSV is
written to disk exactly like a worker would produce, build_trial_record()
reads it through the real metrics.py, the record is appended to a real
TrialHistory, and StopPolicyEvaluator gates each step. runner.py is not
included since it is still in progress elsewhere; everything downstream
of "the worker already ran and wrote a metrics file" is exercised for
real, with no mocking of any project code.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from black_box_optimizer.history import TrialHistory
from black_box_optimizer.models import (
    AlgorithmSpec,
    Direction,
    Objective,
    OptimizationContract,
    ParameterDefinition,
    ParameterKind,
    StopPolicy,
)
from black_box_optimizer.records import build_trial_record
from black_box_optimizer.search.registry import create_algorithm
from black_box_optimizer.stop_policy import StopPolicyEvaluator


def make_contract() -> OptimizationContract:
    """A small contract, small enough to run several trials quickly."""
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.0001, 0.1
            ),
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8, 16, 32)
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


class OneTrialSliceTests(unittest.TestCase):
    """Run real trials through metrics, records, history, search, and
    stop_policy together, without mocking any of our own code."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)

    def write_metrics(
        self, trial_id: int, accuracy: float, loss: float
    ) -> Path:
        """Write a metrics CSV exactly like a worker would produce one."""
        path = self.directory / f"trial_{trial_id}.csv"
        path.write_text(f"accuracy,loss\n{accuracy},{loss}\n")
        return path

    def test_single_trial_slice(self) -> None:
        contract = make_contract()
        history = TrialHistory()
        stop_eval = StopPolicyEvaluator(StopPolicy(max_trials=3))
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=42)
        )

        gate = stop_eval.before_trial(history.snapshot())
        self.assertTrue(gate.continue_execution)

        proposal = algorithm.propose(contract, history.snapshot())
        self.assertEqual(proposal.status, "candidate")

        metrics_path = self.write_metrics(0, accuracy=0.91, loss=0.15)
        execution_result = {
            "execution_status": "completed",
            "runtime_seconds": 1.2,
            "exit_code": 0,
            "timed_out": False,
            "error_message": None,
        }
        record = build_trial_record(
            proposal.candidate, 0, metrics_path, execution_result
        )
        self.assertEqual(record.metrics_status, "valid")
        self.assertEqual(record.metrics["accuracy"], 0.91)
        self.assertEqual(record.parameters, proposal.candidate.parameters)

        history.append(record)
        self.assertEqual(len(history.snapshot()), 1)

    def test_multi_trial_run_grows_history_and_avoids_repeats(self) -> None:
        contract = make_contract()
        history = TrialHistory()
        stop_eval = StopPolicyEvaluator(StopPolicy(max_trials=3))
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=42)
        )

        seen_parameter_sets = []
        for trial_id in range(3):
            gate = stop_eval.before_trial(history.snapshot())
            self.assertTrue(gate.continue_execution)

            proposal = algorithm.propose(contract, history.snapshot())
            self.assertEqual(proposal.status, "candidate")
            seen_parameter_sets.append(proposal.candidate.parameters)

            metrics_path = self.write_metrics(
                trial_id, accuracy=0.8 + trial_id * 0.01, loss=0.2
            )
            execution_result = {
                "execution_status": "completed",
                "runtime_seconds": 1.0,
                "exit_code": 0,
                "timed_out": False,
                "error_message": None,
            }
            record = build_trial_record(
                proposal.candidate, trial_id, metrics_path, execution_result
            )
            history.append(record)

        self.assertEqual(len(history.snapshot()), 3)
        # RandomSearch must never propose the exact same parameters twice.
        self.assertEqual(
            len(seen_parameter_sets),
            len({tuple(sorted(p.items())) for p in seen_parameter_sets}),
        )

        # The policy should now refuse a 4th trial: max_trials was reached.
        final_gate = stop_eval.after_trial(history.snapshot())
        self.assertFalse(final_gate.continue_execution)
        self.assertEqual(final_gate.termination_reason, "maximum_trials")

    def test_missing_metrics_file_flows_through_as_missing_status(self) -> None:
        # Simulates a worker that ran but never wrote a metrics file --
        # the record should reflect that honestly, not crash the slice.
        contract = make_contract()
        history = TrialHistory()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=1)
        )

        proposal = algorithm.propose(contract, history.snapshot())
        missing_path = self.directory / "never_written.csv"
        execution_result = {
            "execution_status": "process_failed",
            "runtime_seconds": 0.5,
            "exit_code": 1,
            "timed_out": False,
            "error_message": "worker exited with code 1",
        }
        record = build_trial_record(
            proposal.candidate, 0, missing_path, execution_result
        )

        self.assertEqual(record.metrics_status, "missing")
        self.assertEqual(record.metrics, {})
        self.assertEqual(record.execution_status, "process_failed")

        # A failed trial still counts as real evidence and must be
        # recorded, not silently dropped.
        history.append(record)
        self.assertEqual(len(history.snapshot()), 1)


if __name__ == "__main__":
    unittest.main()
