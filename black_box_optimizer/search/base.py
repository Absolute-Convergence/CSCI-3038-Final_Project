"""The shared search interface and the result from one proposal attempt."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from black_box_optimizer.models import (
    CandidateConfiguration,
    OptimizationContract,
)
from black_box_optimizer.records import TrialRecord

# These are the only results a search algorithm can give us
ProposalStatus = Literal[
    "candidate",
    "search_exhausted",
    "proposal_failed",
]

# Keep this lined up with the ProposalStatus values above
_VALID_PROPOSAL_STATUSES = (
    "candidate",
    "search_exhausted",
    "proposal_failed",
)


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """
    What happened when a search algorithm tried to propose a candidate.

    A successful result carries a candidate and the other two do not.
    """

    status: ProposalStatus
    candidate: CandidateConfiguration | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Make sure the result does not contradict itself."""
        if self.status not in _VALID_PROPOSAL_STATUSES:
            valid = sorted(_VALID_PROPOSAL_STATUSES)
            raise ValueError(f"status must be one of {valid}")

        # Only a successful proposal should have a candidate attached
        if self.status == "candidate" and self.candidate is None:
            raise ValueError(
                "candidate is required when status is 'candidate'"
            )

        if self.status != "candidate" and self.candidate is not None:
            raise ValueError(
                "candidate must be None unless status is 'candidate'"
            )

        if self.candidate is not None and not isinstance(
            self.candidate,
            CandidateConfiguration,
        ):
            raise TypeError(
                "candidate must be a CandidateConfiguration or None"
            )

        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None")


class SearchAlgorithm(Protocol):
    """The shape every search algorithm in the project needs to follow."""

    def propose(
        self,
        contract: OptimizationContract,
        history: Sequence[TrialRecord],
    ) -> ProposalResult:
        ...