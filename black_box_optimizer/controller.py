"""Deterministic lifecycle control for one sequential optimization run.

Application composition finishes before this controller is constructed. The
controller coordinates search, stopping, worker execution, evidence recording,
checkpointing, result construction, and Reporter delegation without owning the
implementation of those collaborators.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from black_box_optimizer import runner
from black_box_optimizer.candidate_validation import (
    CandidateValidationError,
    validate_candidate,
)
from black_box_optimizer.history import TrialHistory
from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
    WorkerSpec,
)
from black_box_optimizer.persistence import CheckpointError, RunDirectory
from black_box_optimizer.records import TrialRecord, build_trial_record
from black_box_optimizer.reporting import ReportingError, ResultReporter
from black_box_optimizer.results import (
    OptimizationResult,
    build_optimization_result,
)
from black_box_optimizer.search.base import SearchAlgorithm
from black_box_optimizer.stop_policy import (
    StopPolicyEvaluator,
    TerminationReason,
)


class ControllerState(StrEnum):
    """
    States used by the controller lifecycle.

    Initialization is an approved, separate application-composition concern.
    This lifecycle state machine therefore begins at SELECTING.
    """

    SELECTING = "selecting"
    GATING = "gating"
    EXECUTING = "executing"
    RECORDING = "recording"
    EVALUATING = "evaluating"
    FINALIZING = "finalizing"
    FAILED = "failed"
    STOPPED = "stopped"


# Runner owns interrupts while a child is active and returns a cancelled
# observation. If that private contract is violated, or interruption lands
# during evidence persistence, the controller must not fabricate completion.
_UNSAFE_TO_CANCEL_STATES = (
    ControllerState.EXECUTING,
    ControllerState.RECORDING,
)


class ApplicationController:
    """Governs one optimization run's worker-authorization lifecycle."""

    def __init__(
        self,
        contract: OptimizationContract,
        algorithm: SearchAlgorithm,
        stop_policy: StopPolicyEvaluator,
        worker_spec: WorkerSpec,
        run_directory: RunDirectory,
        reporter: ResultReporter,
    ) -> None:
        self._contract = contract
        self._algorithm = algorithm
        self._stop_policy = stop_policy
        self._worker_spec = worker_spec
        self._run_directory = run_directory
        self._reporter = reporter

        self._history = TrialHistory()
        self._next_trial_id = 0

        # The application layer completes initialization before construction.
        self.state: ControllerState = ControllerState.SELECTING
        self.termination_reason: TerminationReason | None = None
        self.result: OptimizationResult | None = None
        self._reported = False

    @property
    def history(self) -> tuple[TrialRecord, ...]:
        """A read-only snapshot of every trial recorded so far."""
        return self._history.snapshot()

    def run(self) -> OptimizationResult:
        """
        Run the state machine until the optimization is finished.

        Return the authoritative immutable result after final reporting.
        """
        candidate: CandidateConfiguration | None = None
        metrics_path: Path | None = None
        execution_result: dict[str, object] | None = None

        try:
            while True:
                if self.state is ControllerState.SELECTING:
                    proposal = self._algorithm.propose(
                        self._contract, self._history.snapshot()
                    )
                    if proposal.status == "candidate":
                        try:
                            candidate = validate_candidate(
                                proposal.candidate,
                                self._contract,
                            )
                        except CandidateValidationError:
                            self.termination_reason = "fatal_error"
                            self.state = ControllerState.FAILED
                        else:
                            self.state = ControllerState.GATING
                    elif proposal.status == "search_exhausted":
                        self.termination_reason = "search_exhausted"
                        self.state = ControllerState.FINALIZING
                    else:
                        # proposal_failed means the search algorithm could
                        # not produce a candidate or a clean exhaustion signal
                        # the controller cannot recover from that
                        self.termination_reason = "fatal_error"
                        self.state = ControllerState.FAILED

                elif self.state is ControllerState.GATING:
                    decision = self._stop_policy.before_trial(
                        self._history.snapshot()
                    )
                    if decision.continue_execution:
                        self.state = ControllerState.EXECUTING
                    else:
                        self.termination_reason = decision.termination_reason
                        self.state = ControllerState.FINALIZING

                elif self.state is ControllerState.EXECUTING:
                    if candidate is None:
                        raise RuntimeError(
                            "EXECUTING reached without a candidate."
                        )

                    metrics_path = self._run_directory.metrics_path(
                        self._next_trial_id
                    )

                    execution_result = runner.execute(
                        self._worker_spec, candidate, metrics_path
                    )
                    self.state = ControllerState.RECORDING

                elif self.state is ControllerState.RECORDING:
                    if candidate is None:
                        raise RuntimeError(
                            "RECORDING reached without a candidate."
                        )
                    if metrics_path is None:
                        raise RuntimeError(
                            "RECORDING reached without a metrics path."
                        )
                    if execution_result is None:
                        raise RuntimeError(
                            "RECORDING reached without an execution result."
                        )

                    record = build_trial_record(
                        candidate,
                        self._next_trial_id,
                        metrics_path,
                        execution_result,
                    )
                    self._history.append(record)
                    self._next_trial_id += 1
                    was_cancelled = record.execution_status == "cancelled"
                    stdout = execution_result.get("stdout", "")
                    stderr = execution_result.get("stderr", "")

                    # Cleared so a future bug cannot silently reuse stale
                    # trial data from the previous worker
                    candidate = None
                    metrics_path = None
                    execution_result = None

                    try:
                        self._run_directory.write_diagnostics(
                            record.trial_id,
                            stdout,
                            stderr,
                        )
                        self._run_directory.checkpoint(
                            self._history.snapshot(), self._contract
                        )
                    except CheckpointError:
                        # TDS section 10 3 specifically treats checkpoint
                        # failure as fatal but keeps the in memory record
                        # so this one expected failure becomes a controller
                        # state instead of bubbling out like a programming bug
                        self.termination_reason = "fatal_error"
                        self.state = ControllerState.FAILED
                        continue

                    if was_cancelled:
                        self.termination_reason = "user_cancelled"
                        self.state = ControllerState.FINALIZING
                    else:
                        self.state = ControllerState.EVALUATING

                elif self.state is ControllerState.EVALUATING:
                    decision = self._stop_policy.after_trial(
                        self._history.snapshot()
                    )
                    if decision.continue_execution:
                        self.state = ControllerState.SELECTING
                    else:
                        self.termination_reason = decision.termination_reason
                        self.state = ControllerState.FINALIZING

                elif self.state is ControllerState.FAILED:
                    # FAILED is only for fatal lifecycle outcomes defined by
                    # the TDS and not for arbitrary programming exceptions
                    self.termination_reason = (
                        self.termination_reason or "fatal_error"
                    )
                    self.state = ControllerState.FINALIZING

                elif self.state is ControllerState.FINALIZING:
                    self._finalize()
                    self.state = ControllerState.STOPPED

                elif self.state is ControllerState.STOPPED:
                    if self.result is None:
                        raise RuntimeError("STOPPED reached without a result")
                    return self.result

                else:
                    # Every ControllerState should have an explicit branch
                    # so a future state cannot create a silent infinite loop
                    raise RuntimeError(
                        f"Unhandled controller state: {self.state!r}"
                    )

        except KeyboardInterrupt:
            # Prelaunch cancellation is safe to finalize normally. Runner owns
            # child shutdown; RECORDING may be inside an evidence commit.
            if self.state in _UNSAFE_TO_CANCEL_STATES:
                raise

            self.termination_reason = "user_cancelled"
            self.state = ControllerState.FINALIZING
            self._finalize()
            self.state = ControllerState.STOPPED
            if self.result is None:
                raise RuntimeError("STOPPED reached without a result")
            return self.result

    def _finalize(self) -> None:
        """Build and report the authoritative result exactly once."""
        if self._reported:
            return
        if self.termination_reason is None:
            raise RuntimeError("FINALIZING requires a termination reason")

        if (
            self.result is None
            or self.result.termination_reason != self.termination_reason
        ):
            self.result = build_optimization_result(
                self._history.snapshot(),
                self._contract,
                self.termination_reason,
            )

        try:
            self._reporter.write(self.result)
        except ReportingError:
            self.termination_reason = "fatal_error"
            self.result = build_optimization_result(
                self._history.snapshot(),
                self._contract,
                self.termination_reason,
            )
            self.state = ControllerState.FAILED
            raise

        self._reported = True
