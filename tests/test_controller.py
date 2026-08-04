"""Focused lifecycle and finalization tests for ApplicationController."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from black_box_optimizer.controller import (
    ApplicationController,
    ControllerState,
)
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
from black_box_optimizer.persistence import (
    CheckpointError,
    create_run_directory,
)
from black_box_optimizer.reporting import ReportingError
from black_box_optimizer.search.base import ProposalResult, SearchAlgorithm
from black_box_optimizer.search.registry import create_algorithm
from black_box_optimizer.stop_policy import StopDecision, StopPolicyEvaluator


def make_contract() -> OptimizationContract:
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "learning_rate", ParameterKind.FLOAT, 0.01, 0.1
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


def make_finite_contract() -> OptimizationContract:
    # No FLOAT parameter here because even one continuous parameter makes
    # RandomSearch treat the space as unbounded and therefore inexhaustible
    return OptimizationContract(
        parameters=(
            ParameterDefinition(
                "batch_size", ParameterKind.CATEGORICAL, choices=(8,)
            ),
        ),
        objectives=(
            Objective("accuracy", Direction.MAXIMIZE),
            Objective("loss", Direction.MINIMIZE),
        ),
    )


def make_worker_spec() -> WorkerSpec:
    # This is never actually launched because runner.execute is faked
    return WorkerSpec(
        command=("does-not-matter",),
        metrics_argument="--metrics-out",
        timeout_seconds=30.0,
    )


class _AlwaysFailingSearch:
    """A SearchAlgorithm stand-in that can never produce a candidate."""

    def propose(self, contract, history):
        return ProposalResult(status="proposal_failed", reason="stalled")


class _InvalidCandidateSearch:
    """Proposes a candidate outside the declared parameter domain."""

    def propose(self, contract, history):
        return ProposalResult(
            status="candidate",
            candidate=CandidateConfiguration(
                parameters={"learning_rate": 5.0, "batch_size": 8}
            ),
        )


class _RecordingReporter:
    def __init__(self) -> None:
        self.results = []

    def write(self, result) -> None:
        self.results.append(result)


class _FailingReporter:
    def write(self, result) -> None:
        raise ReportingError("report failed")


class _InterruptOnceReporter:
    def __init__(self) -> None:
        self.results = []

    def write(self, result) -> None:
        self.results.append(result)
        if len(self.results) == 1:
            raise KeyboardInterrupt


class _InterruptedBeforeLaunchSearch:
    """Simulates Ctrl+C arriving while the controller is still selecting."""

    def propose(self, contract, history):
        raise KeyboardInterrupt


class _AlwaysStoppingPolicy:
    """A StopPolicyEvaluator stand-in that refuses every proposed trial."""

    def before_trial(self, history):
        return StopDecision(
            continue_execution=False, termination_reason="maximum_trials"
        )

    def after_trial(self, history):
        return StopDecision(
            continue_execution=False, termination_reason="maximum_trials"
        )


def _write_valid_metrics(
    path: Path, accuracy: float = 0.9, loss: float = 0.1
) -> None:
    path.write_text(f"accuracy,loss\n{accuracy},{loss}\n")


def _fake_completed_execute(metrics_by_path=None, accuracy=0.9, loss=0.1):
    """
    Build a fake runner.execute() that always finishes successfully.

    It still writes a totally real parseable metrics file so metrics.py
    and records.py are tested through their actual implementations.
    """

    def _execute(worker_spec, candidate, metrics_path):
        _write_valid_metrics(Path(metrics_path), accuracy=accuracy, loss=loss)
        if metrics_by_path is not None:
            metrics_by_path.append(Path(metrics_path))
        return {
            "runtime_seconds": 0.5,
            "exit_code": 0,
            "timed_out": False,
            "execution_status": "completed",
            "error_message": None,
        }

    return _execute


class ControllerHappyPathTests(unittest.TestCase):
    """The normal propose, gate, execute, record, evaluate, repeat loop."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = create_run_directory(self._tmpdir.name)
        self.reporter = _RecordingReporter()

    def test_reaches_finalizing_with_maximum_trials(self) -> None:
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=1)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=2))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(),
        ):
            result = controller.run()

        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertEqual(controller.termination_reason, "maximum_trials")
        self.assertEqual(len(controller.history), 2)
        self.assertIs(result, controller.result)
        self.assertEqual(result.status, "completed")

        for expected_id, record in enumerate(controller.history):
            self.assertEqual(record.trial_id, expected_id)
            self.assertEqual(record.execution_status, "completed")
            self.assertEqual(record.metrics_status, "valid")
            self.assertEqual(record.metrics["accuracy"], 0.9)
            self.assertIn("learning_rate", record.parameters)

    def test_metrics_path_is_unique_per_trial(self) -> None:
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=2)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=3))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        seen_paths: list[Path] = []
        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(metrics_by_path=seen_paths),
        ):
            controller.run()

        self.assertEqual(len(seen_paths), 3)
        self.assertEqual(len(set(seen_paths)), 3)

    def test_recording_happens_even_when_execution_fails(self) -> None:
        # TDS section 5 2 says every authorized attempt reaches RECORDING
        # even if launch execution metrics or cancellation fails
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=3)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=1))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        def _failing_execute(worker_spec, candidate, metrics_path):
            # No metrics file here because a real failed process may never
            # produce one and records.py needs to classify that correctly
            return {
                "runtime_seconds": 0.1,
                "exit_code": 1,
                "timed_out": False,
                "execution_status": "process_failed",
                "error_message": None,
            }

        with patch(
            "black_box_optimizer.runner.execute", side_effect=_failing_execute
        ):
            result = controller.run()

        self.assertEqual(len(controller.history), 1)
        record = controller.history[0]
        self.assertEqual(record.execution_status, "process_failed")
        self.assertEqual(record.metrics_status, "missing")
        self.assertEqual(result.status, "no_eligible_trials")


class ControllerTerminationReasonTests(unittest.TestCase):
    """Every currently reachable path into FINALIZING and its reason."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = create_run_directory(self._tmpdir.name)
        self.reporter = _RecordingReporter()

    def test_search_exhausted_after_one_real_trial(self) -> None:
        # This domain has exactly one possible candidate and no FLOAT
        # parameter so the real RandomSearch can genuinely exhaust it
        contract = make_finite_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=4)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(),
        ):
            result = controller.run()

        self.assertEqual(controller.termination_reason, "search_exhausted")
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(result.status, "completed")

    def test_checkpoint_failure_finalizes_with_fatal_error(self) -> None:
        # TDS section 10 3 says checkpoint failure is fatal but the record
        # already stored in memory must survive and CheckpointError must not
        # escape directly from run
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=8)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(),
        ):
            with patch(
                "black_box_optimizer.controller.RunDirectory.checkpoint",
                side_effect=CheckpointError("disk full"),
            ):
                result = controller.run()

        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertEqual(controller.termination_reason, "fatal_error")
        self.assertEqual(result.status, "failed")

        # The record is appended before checkpointing so the failed disk
        # write should not erase the real in memory history
        self.assertEqual(len(controller.history), 1)

    def test_trial_directory_failure_preserves_record_and_finalizes(
        self,
    ) -> None:
        controller = ApplicationController(
            make_contract(),
            create_algorithm(AlgorithmSpec("random_search", seed=8)),
            StopPolicyEvaluator(StopPolicy(max_trials=1)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )
        real_trial_directory = self.run_directory.trial_directory
        calls = 0

        def fail_second_trial_directory(trial_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise CheckpointError("permission denied")
            return real_trial_directory(trial_id)

        with patch.object(
            self.run_directory,
            "trial_directory",
            side_effect=fail_second_trial_directory,
        ):
            with patch(
                "black_box_optimizer.runner.execute",
                side_effect=_fake_completed_execute(),
            ):
                result = controller.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "fatal_error")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0].execution_status, "completed")

    def test_proposal_failed_becomes_fatal_error_via_failed_state(self) -> None:
        contract = make_contract()
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            _AlwaysFailingSearch(),
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        result = controller.run()

        self.assertEqual(controller.termination_reason, "fatal_error")
        self.assertEqual(len(controller.history), 0)
        self.assertEqual(result.status, "failed")

    def test_gating_stop_finalizes_before_any_execution(self) -> None:
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=6)
        )
        controller = ApplicationController(
            contract,
            algorithm,
            _AlwaysStoppingPolicy(),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        with patch("black_box_optimizer.runner.execute") as fake_execute:
            result = controller.run()
            fake_execute.assert_not_called()

        self.assertEqual(controller.termination_reason, "maximum_trials")
        self.assertEqual(len(controller.history), 0)
        self.assertEqual(result.status, "no_eligible_trials")

    def test_invalid_candidate_is_fatal_and_never_launched(self) -> None:
        controller = ApplicationController(
            make_contract(),
            _InvalidCandidateSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        with patch("black_box_optimizer.runner.execute") as fake_execute:
            result = controller.run()

        fake_execute.assert_not_called()
        self.assertEqual(controller.termination_reason, "fatal_error")
        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertEqual(controller.history, ())
        self.assertEqual(result.status, "failed")

    def test_pre_launch_interrupt_becomes_user_cancelled(self) -> None:
        contract = make_contract()
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            _InterruptedBeforeLaunchSearch(),
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        result = controller.run()

        self.assertEqual(controller.termination_reason, "user_cancelled")
        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertEqual(len(controller.history), 0)
        self.assertEqual(result.status, "cancelled")

    def test_runner_cancellation_records_attempt_and_diagnostics(self) -> None:
        contract = make_contract()
        controller = ApplicationController(
            contract,
            create_algorithm(AlgorithmSpec(name="random_search", seed=9)),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )
        cancelled_observation = {
            "runtime_seconds": 0.2,
            "exit_code": -15,
            "timed_out": False,
            "execution_status": "cancelled",
            "error_message": "Worker cancelled by user",
            "stdout": "partial output",
            "stderr": "shutdown detail",
        }

        with patch(
            "black_box_optimizer.runner.execute",
            return_value=cancelled_observation,
        ):
            result = controller.run()

        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertEqual(controller.termination_reason, "user_cancelled")
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0].execution_status, "cancelled")
        trial_directory = self.run_directory.trial_directory(0)
        self.assertEqual(
            (trial_directory / "stdout.txt").read_text(encoding="utf-8"),
            "partial output",
        )
        self.assertEqual(
            (trial_directory / "stderr.txt").read_text(encoding="utf-8"),
            "shutdown detail",
        )

    def test_unexpected_runner_interrupt_is_not_swallowed(self) -> None:
        contract = make_contract()
        algorithm = create_algorithm(
            AlgorithmSpec(name="random_search", seed=7)
        )
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            algorithm,
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        def _interrupted_execute(worker_spec, candidate, metrics_path):
            raise KeyboardInterrupt

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_interrupted_execute,
        ):
            # Runner owns child shutdown and normally returns a cancelled
            # observation. A runner that violates that boundary stays loud.
            with self.assertRaises(KeyboardInterrupt):
                controller.run()

        self.assertEqual(controller.state, ControllerState.EXECUTING)
        self.assertEqual(len(controller.history), 0)


class ControllerInternalErrorTests(unittest.TestCase):
    """Every authorized attempt survives internal collaborator failures."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = create_run_directory(self._tmpdir.name)
        self.reporter = _RecordingReporter()

    def make_controller(self) -> ApplicationController:
        return ApplicationController(
            make_contract(),
            create_algorithm(AlgorithmSpec("random_search", seed=9)),
            StopPolicyEvaluator(StopPolicy(max_trials=1)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

    def test_unexpected_runner_error_is_recorded_and_finalized(self) -> None:
        controller = self.make_controller()

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=RuntimeError("runner contract broke"),
        ):
            result = controller.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "fatal_error")
        self.assertEqual(len(result.history), 1)
        record = result.history[0]
        self.assertEqual(record.execution_status, "internal_error")
        self.assertEqual(record.metrics_status, "missing")
        self.assertIn("runner contract broke", record.error_message)
        self.assertTrue(
            (self.run_directory.path / "history.csv").is_file()
        )
        history_text = (self.run_directory.path / "history.csv").read_text()
        self.assertIn("internal_error", history_text)
        self.assertEqual(result.pareto_count, 0)
        self.assertIs(controller.run(), result)

    def test_unexpected_record_factory_error_is_recorded(self) -> None:
        controller = self.make_controller()

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(),
        ):
            with patch(
                "black_box_optimizer.controller.build_trial_record",
                side_effect=KeyError("runtime_seconds"),
            ):
                result = controller.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.history), 1)
        record = result.history[0]
        self.assertEqual(record.execution_status, "internal_error")
        self.assertIn("KeyError", record.error_message)
        self.assertIn("runtime_seconds", record.error_message)


class ControllerFinalizeTests(unittest.TestCase):
    """Finalization builds one authoritative result and delegates reporting."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = create_run_directory(self._tmpdir.name)
        self.reporter = _RecordingReporter()

    def test_finalization_is_idempotent_and_reports_exact_result(self) -> None:
        contract = make_contract()
        stop_policy = StopPolicyEvaluator(StopPolicy(max_trials=5))
        controller = ApplicationController(
            contract,
            _AlwaysFailingSearch(),
            stop_policy,
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        result = controller.run()
        repeated = controller.run()

        self.assertIs(repeated, result)
        self.assertEqual(self.reporter.results, [result])
        self.assertEqual(controller.state, ControllerState.STOPPED)

    def test_reporting_failure_is_fatal_and_retains_result(self) -> None:
        controller = ApplicationController(
            make_contract(),
            _AlwaysFailingSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            _FailingReporter(),
        )

        with self.assertRaisesRegex(ReportingError, "report failed"):
            controller.run()

        self.assertEqual(controller.state, ControllerState.FAILED)
        self.assertEqual(controller.termination_reason, "fatal_error")
        self.assertIsNotNone(controller.result)
        self.assertEqual(controller.result.status, "failed")

    def test_finalization_interrupt_preserves_result_and_retries(self) -> None:
        reporter = _InterruptOnceReporter()
        controller = ApplicationController(
            make_contract(),
            create_algorithm(AlgorithmSpec("random_search", seed=11)),
            StopPolicyEvaluator(StopPolicy(max_trials=1)),
            make_worker_spec(),
            self.run_directory,
            reporter,
        )

        with patch(
            "black_box_optimizer.runner.execute",
            side_effect=_fake_completed_execute(),
        ):
            with self.assertRaises(KeyboardInterrupt):
                controller.run()

        first_result = controller.result
        self.assertIsNotNone(first_result)
        self.assertEqual(first_result.status, "completed")
        self.assertEqual(first_result.termination_reason, "maximum_trials")
        self.assertEqual(controller.state, ControllerState.FINALIZING)
        self.assertEqual(reporter.results, [first_result])

        repeated = controller.run()

        self.assertIs(repeated, first_result)
        self.assertEqual(reporter.results, [first_result, first_result])
        self.assertEqual(controller.state, ControllerState.STOPPED)

    def test_calling_finalize_again_directly_is_a_no_op(self) -> None:
        # run() calling itself twice can't reach this: the second run()
        # short-circuits at the STOPPED branch before ever calling
        # _finalize() again. Calling the private method directly is the
        # only way to exercise _finalize()'s own _reported guard.
        controller = ApplicationController(
            make_contract(),
            _AlwaysFailingSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        result = controller.run()
        controller._finalize()

        self.assertIs(controller.result, result)
        self.assertEqual(self.reporter.results, [result])

    def test_finalize_retries_the_write_without_rebuilding_a_matching_result(
        self,
    ) -> None:
        # Simulates retrying delivery after some external code resets
        # _reported (e.g. after catching a ReportingError and wanting
        # another attempt) -- the already-correct result must not be
        # rebuilt from scratch, only re-delivered.
        controller = ApplicationController(
            make_contract(),
            _AlwaysFailingSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

        result = controller.run()
        controller._reported = False
        controller._finalize()

        self.assertIs(controller.result, result)
        self.assertEqual(self.reporter.results, [result, result])


class ControllerInvariantGuardTests(unittest.TestCase):
    """Defensive guards against internally-corrupted state.

    None of these are reachable through the controller's real public
    lifecycle -- each one requires forcing `state`/`_reported` directly to
    simulate a hypothetical future bug, exactly the kind of "this should
    never happen" guard that's worth pinning down with a real test rather
    than trusting it silently.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.run_directory = create_run_directory(self._tmpdir.name)
        self.reporter = _RecordingReporter()

    def _make_controller(self) -> ApplicationController:
        return ApplicationController(
            make_contract(),
            _AlwaysFailingSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )

    def test_executing_without_a_candidate_raises(self) -> None:
        controller = self._make_controller()
        controller.state = ControllerState.EXECUTING

        with self.assertRaisesRegex(RuntimeError, "without a candidate"):
            controller.run()

    def test_recording_without_a_candidate_raises(self) -> None:
        controller = self._make_controller()
        controller.state = ControllerState.RECORDING

        with self.assertRaisesRegex(RuntimeError, "without a candidate"):
            controller.run()

    def test_stopped_without_a_result_raises(self) -> None:
        controller = self._make_controller()
        controller.state = ControllerState.STOPPED

        with self.assertRaisesRegex(RuntimeError, "without a result"):
            controller.run()

    def test_unhandled_state_raises(self) -> None:
        controller = self._make_controller()
        controller.state = "not_a_real_state"

        with self.assertRaisesRegex(RuntimeError, "Unhandled controller state"):
            controller.run()

    def test_stopped_without_a_result_during_interrupt_raises(self) -> None:
        # Forces the exact inconsistency this guard exists for: _reported
        # already true (so _finalize() early-returns without building a
        # result) but no result was ever actually produced.
        controller = ApplicationController(
            make_contract(),
            _InterruptedBeforeLaunchSearch(),
            StopPolicyEvaluator(StopPolicy(max_trials=5)),
            make_worker_spec(),
            self.run_directory,
            self.reporter,
        )
        controller._reported = True

        with self.assertRaisesRegex(RuntimeError, "without a result"):
            controller.run()

    def test_finalize_without_a_termination_reason_raises(self) -> None:
        controller = self._make_controller()

        with self.assertRaisesRegex(
            RuntimeError, "requires a termination reason"
        ):
            controller._finalize()


if __name__ == "__main__":
    unittest.main()
