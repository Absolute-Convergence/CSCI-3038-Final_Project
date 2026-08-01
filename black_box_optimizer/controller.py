"""
controller.py

The Controller! This is the deterministic finite-state lifecycle
governor from section 5 of the TDS. Its whole job is deciding what
state comes next and making sure only one worker is ever authorized at
a time.

It doesn’t actually know anything about optimization itself. Candidate
generation belongs to the search algorithm, execution budgeting belongs
to StopPolicy, actually running the worker belongs to runner.execute(),
and turning that run into a TrialRecord belongs to
build_trial_record(). The controller just coordinates everybody else.

It also doesn’t load JSON (that’s config_loader.py’s deal). It
expects already built config objects and also an already created
RunDirectory to handle any metrics files and history checkpoints.

KNOWN GAP!!! FINALIZING isn’t actually finished yet. According to the
spec it’s supposed to derive a ParetoFront, build an
OptimizationResult, then hand everything off to a Reporter. The catch
is… results.py and reporting.py don’t exist yet, and pareto.py only
implements is_eligible(), not the actual Pareto sweep. So _finalize()
is an intentional seam for now it raises NotImplementedError.

(The good news is that by the time _finalize() is done, the other
important stuff is totally correct. The controller state,
termination_reason, and the full TrialHistory have all been built.
The only thing missing is that final Pareto/report step.)

LAST KNOWN GAP!!! Mid-worker cancellation isn’t implemented yet. The
spec says a KeyboardInterrupt during a worker should terminate the child
process and still record a cancelled trial. Right now only pre-launch
cancellation is handled for real. If Ctrl+C lands while the worker is
running, the interrupt is raised again instead of pretending everything
is fine. Safely killing the child would require changing runner.pys
blocking subprocess.run() call, and that’s a contract change so best to
discuss amongst I suppose.

Final last note! Unexpected exceptions (anything besides
KeyboardInterrupt) from runner.execute(), build_trial_record(), or the
search/stop-policy collaborators are allowed to bubble up on purpose.
Those are programming bugs, not normal controller states, because I
fear that quietly routing them through FAILED/FINALIZING would just
make debugging a huge big massive pain so just heads up.
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
from black_box_optimizer.persistence import RunDirectory
from black_box_optimizer.records import TrialRecord, build_trial_record
from black_box_optimizer.search.base import SearchAlgorithm
from black_box_optimizer.stop_policy import (
    StopPolicyEvaluator,
    TerminationReason,
)


class ControllerState(StrEnum):
    """States outlined in the state model."""

    SELECTING = "selecting"
    GATING = "gating"
    EXECUTING = "executing"
    RECORDING = "recording"
    EVALUATING = "evaluating"
    FINALIZING = "finalizing"
    FAILED = "failed"
    STOPPED = "stopped"


# EXECUTING: the child process may still be alive, nothing to record yet.
# RECORDING: runner.execute() has already returned, but appending to
# history and advancing the trial counter isn't atomic -- an interrupt
# partway through could leave that commit half-done.
# A KeyboardInterrupt caught in either is raised again instead of being
# treated as a clean pre-launch cancellation.
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

        self.state: ControllerState = ControllerState.SELECTING
        self.termination_reason: TerminationReason | None = None

    @property
    def history(self) -> tuple[TrialRecord, ...]:
        """A read-only snapshot of every trial recorded so far."""
        return self._history.snapshot()

    def run(self) -> None:
        """
        Run the state machine until FINALIZING is reached.

        Always raises NotImplementedError once FINALIZING is reached --
        see the module docstring's KNOWN GAP notes. self.state,
        self.termination_reason, and self.history are already correct
        by the time that happens.

        This is meant to be replaced whenever! See known gaps
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
                        # proposal_failed: the search algorithm could not
                        # produce a usable candidate or a clean exhaustion
                        # signal, which is a case the run cannot recover
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
                    self._run_directory.checkpoint(
                        self._history.snapshot(), self._contract
                    )

                    # Cleared so a future bug can't silently reuse a stale
                    # candidate/path/result from a previous trial.
                    candidate = None
                    metrics_path = None
                    execution_result = None

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
                    self.termination_reason = (
                        self.termination_reason or "fatal_error"
                    )
                    self.state = ControllerState.FINALIZING

                elif self.state is ControllerState.FINALIZING:
                    self._finalize()
                    # pragma: no cover -- unreachable until finalize exists
                    self.state = ControllerState.STOPPED

                elif self.state is ControllerState.STOPPED:
                    return

                else:
                    raise RuntimeError(
                        f"Unhandled controller state: {self.state!r}"
                    )

        except KeyboardInterrupt:
            if self.state in _UNSAFE_TO_CANCEL_STATES:
                raise
            self.termination_reason = "user_cancelled"
            self.state = ControllerState.FINALIZING
            self._finalize()
            # pragma: no cover -- unreachable until finalize exists
            self.state = ControllerState.STOPPED

    def _finalize(self) -> None:
        """
        Derive the ParetoFront, build an OptimizationResult, and hand it
        to a Reporter

        Intentionally unimplemented as a known gap!
        """
        raise NotImplementedError(
            "FINALIZING requires pareto.py (ParetoFront/OptimizationResult) "
            "and reporting.py (Reporter.write()), neither of which exist "
            f"yet. Reached FINALIZING with termination_reason="
            f"{self.termination_reason!r} and "
            f"{len(self._history.snapshot())} recorded trial(s)."
        )
