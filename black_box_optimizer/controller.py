"""
controller.py

The Controller: the deterministic finite-state lifecycle
governor. It owns state transitions and authorizes at most one
worker execution at a time. It delegates candidate generation
to the search algorithm, execution budgeting to StopPolicy,
subprocess execution to runner.execute(), and record
construction to build_trial_record(). By designm it does not
rank metrics or know worker internals itself.

It does not load JSON (config_loader.py's job) and does not write
anything to disk itself; it accepts already-built configuration objects
and a directory to write per-trial metrics files into.

KNOWN GAP!!! FINALIZING (section 5.1) is supposed to derive a
ParetoFront, construct an OptimizationResult, and hand it to a
Reporter. But since none of pareto.py, results.py, or reporting.py exist yet, so
_finalize() is an intentional, explicit seam: it raises
NotImplementedError. By the time _finalize() runs, state,
termination_reason, and the complete real TrialHistory are already
correct and inspectable -- only the final Pareto/report step is
missing.

ANOTHER KNOWN GAP!!! RECORDING's persistence.checkpoint(history.snapshot()) call
(section 5.4) is omitted, since persistence.py does not exist yet.
This only affects on-disk durability/resume, not correctness -- history
is still fully and correctly tracked in memory.

LAST KNOWN GAP!!! Midworker cancellation (section 5.3's "KeyboardInterrupt
during worker" row -- terminate the child, then record a cancelled
attempt) is not implemented. Only pre-launch cancellation (interrupting
before a worker is authorized) is handled for real; an interrupt that
lands while EXECUTING is in progress is re-raised rather than silently
mishandled, since safely killing the child would mean changing
runner.py's blocking subprocess.run() call, must consult the group for
contract changes such as this!

Unexpected (non-KeyboardInterrupt) exceptions from runner.execute(),
build_trial_record(), or the search/stop-policy collaborators are also
allowed to propagate straight to the caller rather than being caught
and routed into FAILED/FINALIZING. This is deliberate, not an
oversight -- swallowing programming errors into a state transition
would hide real bugs the same way silently swallowing a mid-worker
KeyboardInterrupt would.
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
        metrics_directory: str | Path,
    ) -> None:
        self._contract = contract
        self._algorithm = algorithm
        self._stop_policy = stop_policy
        self._worker_spec = worker_spec
        self._metrics_directory = Path(metrics_directory)
        self._metrics_directory.mkdir(parents=True, exist_ok=True)

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

                    metrics_path = (
                        self._metrics_directory
                        / f"trial_{self._next_trial_id}.csv"
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
