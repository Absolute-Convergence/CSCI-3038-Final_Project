"""Maximum-trial stop decisions, per TDS section 5.3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from black_box_optimizer.models import StopPolicy
from black_box_optimizer.records import TrialRecord

# These are the only termination reasons recognized by the project
TerminationReason = Literal[
    "maximum_trials", "search_exhausted", "user_cancelled", "fatal_error"
]

# Keep this in sync with the TerminationReason values above
_VALID_TERMINATION_REASONS = (
    "maximum_trials", "search_exhausted", "user_cancelled", "fatal_error"
)


@dataclass(frozen=True, slots=True)
class StopDecision:
    """
    Whether the controller should continue running trials.

    If execution stops, termination_reason explains why.
    """

    continue_execution: bool
    termination_reason: TerminationReason | None
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate that the decision is internally consistent."""
        if not isinstance(self.continue_execution, bool):
            raise TypeError("continue_execution must be a bool")

        reason = self.termination_reason
        if reason is not None and reason not in _VALID_TERMINATION_REASONS:
            valid = sorted(_VALID_TERMINATION_REASONS)
            raise ValueError(
                f"termination_reason must be None or one of {valid}"
            )

        # Continuing means there cannot be a termination reason and
        # stopping requires one so the controller can know why execution ended
        if self.continue_execution and reason is not None:
            raise ValueError(
                "continue_execution cannot be True when a termination_reason "
                "is set"
            )
        if not self.continue_execution and reason is None:
            raise ValueError(
                "termination_reason is required when continue_execution is "
                "False"
            )

        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None")


@dataclass(frozen=True, slots=True)
class StopPolicyEvaluator:
    """
    Decides whether another trial may run.

    This evaluator is only responsible for the "maximum_trials"
    termination reason. The remaining termination reasons come from
    other parts of the controller.
    """

    policy: StopPolicy

    def before_trial(self, history: Sequence[TrialRecord]) -> StopDecision:
        """Evaluate the stop policy before launching another worker."""
        return self._evaluate(history)

    def after_trial(self, history: Sequence[TrialRecord]) -> StopDecision:
        """Evaluate the stop policy after a worker finishes."""
        return self._evaluate(history)

    def _evaluate(self, history: Sequence[TrialRecord]) -> StopDecision:
        """Return the stop decision for the current trial history."""
        max_trials = self.policy.max_trials

        # Once the configured limit has been reached, the controller should
        # stop scheduling new trials
        if len(history) >= max_trials:
            return StopDecision(
                continue_execution=False,
                termination_reason="maximum_trials",
                message=f"Reached the maximum of {max_trials} trials.",
            )

        return StopDecision(
            continue_execution=True,
            termination_reason=None,
        )