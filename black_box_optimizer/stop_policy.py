"""Maximum-trial and excessive-failure stop decisions, per TDS section 5.3
and docs/decisions/2026-08-04-repeated-failure-stop-policy-proposal.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from black_box_optimizer.models import OptimizationContract, StopPolicy
from black_box_optimizer.records import TrialRecord

# These are the only termination reasons recognized by the project
TerminationReason = Literal[
    "maximum_trials",
    "search_exhausted",
    "user_cancelled",
    "fatal_error",
    "excessive_failures",
]

# Keep this in sync with the TerminationReason values above
_VALID_TERMINATION_REASONS = (
    "maximum_trials",
    "search_exhausted",
    "user_cancelled",
    "fatal_error",
    "excessive_failures",
)

# A structurally broken config/worker mismatch fails identically on every
# trial regardless of which candidate was tried, so it doesn't need the
# full trial budget to detect -- repeating an uninformative failure just
# wastes the rest of the run. These two constants were calibrated against
# a real, hand-verified binomial simulation (see the proposal doc): 30%
# is an assumed acceptable baseline flakiness rate for an otherwise
# healthy worker, and alpha=0.001 (not the more obvious 0.01) is what
# actually keeps the real false-positive rate near 1% once you account
# for this check running after every single trial instead of once --
# alpha=0.01 checked every trial produced a measured ~6-7% false-positive
# rate in simulation, not the intended 1%.
_BASELINE_ACCEPTABLE_FAILURE_RATE = 0.3
_FAILURE_SIGNIFICANCE_ALPHA = 0.001


def _binomial_tail_probability(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact, no scipy dependency."""
    return sum(
        math.comb(n, i) * (p**i) * ((1 - p) ** (n - i))
        for i in range(k, n + 1)
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

    This evaluator is responsible for the "maximum_trials" termination
    reason, and -- when a contract is supplied -- "excessive_failures".
    The remaining termination reasons come from other parts of the
    controller.

    contract is optional and defaults to None so existing callers that
    only care about max_trials (most of this project's own test suite)
    are unaffected; the excessive-failures check is simply skipped when
    no contract is available to judge trial eligibility against.
    """

    policy: StopPolicy
    contract: OptimizationContract | None = None

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

        excessive_failures = self._excessive_failures(history)
        if excessive_failures is not None:
            return excessive_failures

        return StopDecision(
            continue_execution=True,
            termination_reason=None,
        )

    def _excessive_failures(
        self, history: Sequence[TrialRecord]
    ) -> StopDecision | None:
        """Stop early if failures so far are statistically inconsistent
        with ordinary flakiness -- see the module docstring's proposal
        doc reference for the full reasoning and calibration.
        """
        if self.contract is None or not history:
            return None

        # Local import: pareto.py imports from results.py, which imports
        # TerminationReason from this module -- importing is_eligible at
        # module level here would be circular. results.py's own
        # build_optimization_result() already solves the same problem
        # the same way.
        from black_box_optimizer.pareto import is_eligible

        trial_count = len(history)
        failure_count = sum(
            not is_eligible(record, self.contract) for record in history
        )
        probability = _binomial_tail_probability(
            failure_count, trial_count, _BASELINE_ACCEPTABLE_FAILURE_RATE
        )
        if probability >= _FAILURE_SIGNIFICANCE_ALPHA:
            return None

        return StopDecision(
            continue_execution=False,
            termination_reason="excessive_failures",
            message=(
                f"{failure_count} of {trial_count} trials failed to "
                "produce usable metrics -- statistically inconsistent "
                "with ordinary flakiness, stopping before the "
                "configured trial budget is exhausted."
            ),
        )