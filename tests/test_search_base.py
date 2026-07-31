"""Focused tests for ProposalResult."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from black_box_optimizer.models import CandidateConfiguration
from black_box_optimizer.search.base import ProposalResult


class ProposalResultTests(unittest.TestCase):
    """Verify construction, validation, and immutability of ProposalResult."""

    def make_candidate(self) -> CandidateConfiguration:
        return CandidateConfiguration(parameters={"learning_rate": 0.01})

    # Tests that a "candidate" result requires an actual candidate, and
    # that the other two statuses forbid one.

    def test_candidate_status_requires_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            ProposalResult(status="candidate", candidate=None)

    def test_candidate_status_accepts_candidate(self) -> None:
        result = ProposalResult(
            status="candidate", candidate=self.make_candidate()
        )
        self.assertIsNotNone(result.candidate)

    def test_search_exhausted_forbids_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be None"):
            ProposalResult(
                status="search_exhausted", candidate=self.make_candidate()
            )

    def test_proposal_failed_forbids_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be None"):
            ProposalResult(
                status="proposal_failed", candidate=self.make_candidate()
            )

    # Tests that reject bad field values.

    def test_invalid_status_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "status"):
            ProposalResult(status="banana")

    def test_non_string_reason_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "string"):
            ProposalResult(status="proposal_failed", reason=404)

    def test_wrong_candidate_type_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "CandidateConfiguration"):
            ProposalResult(
                status="candidate",
                candidate={"learning_rate": 0.01},  # type: ignore[arg-type]
            )

    def test_result_is_frozen(self) -> None:
        result = ProposalResult(status="search_exhausted")
        with self.assertRaises(FrozenInstanceError):
            result.status = "proposal_failed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
