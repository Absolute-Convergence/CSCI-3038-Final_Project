"""
controller.py

The Controller! This is the deterministic finite-state lifecycle
governor from TDS section 5. It owns state transitions and makes sure
only one worker is ever authorized at a time.

It doesn't know nuthin about optimization itself. The search algorithm
proposes candidates, StopPolicy decides when to stop, runner.execute()
launches the worker, and build_trial_record() turns the result into a
TrialRecord. The controller just coordinates everybody else.

It also expects already-built configuration objects and an
already-created RunDirectory. Loading JSON and the rest of the setup
happen before the controller is ever constructed.

Keep an eye out (!!!) for the KNOWN DEVIATION and KNOWN GAP notes
throughout the file. They point out the omitted INITIALIZING state,
checkpoint failure handling, mid-worker cancellation, and the
unfinished FINALIZING step while the project waits on pareto.py,
results.py, reporting.py, and their associated stuffs.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from black_box_optimizer import runner
from black_box_optimizer.history import TrialHistory
from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
    WorkerSpec,
)
from black_box_optimizer.persistence import CheckpointError, RunDirectory
from black_box_optimizer.records import TrialRecord, build_trial_record
from black_box_optimizer.search.base import SearchAlgorithm
from black_box_optimizer.stop_policy import (
    StopPolicyEvaluator,
    TerminationReason,
)


class ControllerState(StrEnum):
    """
    States used by the controller lifecycle.

    KNOWN DEVIATION!!! The TDS also includes INITIALIZING, but all of
    that setup happens before ApplicationController is constructed.
    This state machine therefore begins at SELECTING.
    """

    SELECTING = "selecting"
    GATING = "gating"
    EXECUTING = "executing"
    RECORDING = "recording"
    EVALUATING = "evaluating"
    FINALIZING = "finalizing"
    FAILED = "failed"
    STOPPED = "stopped"


# KNOWN GAP
# EXECUTING may still have a living child process
# RECORDING has a finished worker but may be halfway through committing
# the completed trial to history
# An interrupt in either state is raised again because neither can be
# honestly treated as a clean prelaunch cancellation yet
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
    ) -> None:
        self._contract = contract
        self._algorithm = algorithm
        self._stop_policy = stop_policy
        self._worker_spec = worker_spec
        self._run_directory = run_directory

        self._history = TrialHistory()
        self._next_trial_id = 0

        # KNOWN DEVIATION
        # The caller already handled the TDS INITIALIZING work so the
        # controller starts directly at SELECTING
        self.state: ControllerState = ControllerState.SELECTING
        self.termination_reason: TerminationReason | None = None

    @property
    def history(self) -> tuple[TrialRecord, ...]:
        """A read-only snapshot of every trial recorded so far."""
        return self._history.snapshot()

    def run(self) -> None:
        """
        Run the state machine until the optimization is finished.

        KNOWN GAP!!! FINALIZING currently raises NotImplementedError
        because Pareto evaluation and reporting are not implemented yet.
        By the time that happens, state, termination_reason, and history
        are already correct and available for inspection.

        Once _finalize() is implemented, run() will continue into
        STOPPED and return normally.
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
                        candidate = proposal.candidate
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

                    # KNOWN GAP
                    # runner.execute uses a blocking subprocess call so this
                    # controller cannot safely terminate and record a cancelled
                    # child process if Ctrl C lands during execution
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

                    # Cleared so a future bug cannot silently reuse stale
                    # trial data from the previous worker
                    candidate = None
                    metrics_path = None
                    execution_result = None

                    try:
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

                    # Unreachable until the known FINALIZING gap is filled
                    # pragma: no cover
                    self.state = ControllerState.STOPPED

                elif self.state is ControllerState.STOPPED:
                    return

                else:
                    # Every ControllerState should have an explicit branch
                    # so a future state cannot create a silent infinite loop
                    raise RuntimeError(
                        f"Unhandled controller state: {self.state!r}"
                    )

        except KeyboardInterrupt:
            # KNOWN GAP
            # Prelaunch cancellation is safe to finalize normally
            # Midworker or midrecording cancellation is raised again because
            # the controller cannot guarantee a clean child shutdown or commit
            if self.state in _UNSAFE_TO_CANCEL_STATES:
                raise

            self.termination_reason = "user_cancelled"
            self.state = ControllerState.FINALIZING
            self._finalize()

            # Unreachable until the known FINALIZING gap is filled
            # pragma: no cover
            self.state = ControllerState.STOPPED

    def _finalize(self) -> None:
        """
        Derive the ParetoFront, build an OptimizationResult, and hand it
        to a Reporter.

        KNOWN GAP!!! pareto.py does not have the full Pareto sweep yet,
        and results.py and reporting.py do not exist. This method stays
        as an explicit seam until those pieces are ready.
        """
        raise NotImplementedError(
            "FINALIZING requires pareto.py (ParetoFront/OptimizationResult) "
            "and reporting.py (Reporter.write()), neither of which exist "
            f"yet. Reached FINALIZING with termination_reason="
            f"{self.termination_reason!r} and "
            f"{len(self._history.snapshot())} recorded trial(s)."
        )