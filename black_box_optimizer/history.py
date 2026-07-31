"""Append-only TrialHistory, per the interface summary in TDS section 11.1."""

from __future__ import annotations

from black_box_optimizer.records import TrialRecord


class TrialHistory:
    """
    Ordered, append-only collection of TrialRecords for one run.

    Internally mutable so new trials can be appended, but callers only
    ever see read-only tuple snapshots.
    """

    def __init__(self) -> None:
        """Start with an empty history."""
        self._records: list[TrialRecord] = []

    def append(self, record: TrialRecord) -> None:
        """Add one TrialRecord, preserving append-only ordering."""
        # Trial ids should always increase as the optimizer runs so if we see
        # one go backwards or repeat, it means something upstream is wrong
        if self._records and record.trial_id <= self._records[-1].trial_id:
            raise ValueError(
                f"trial_id {record.trial_id} is duplicate or out of order "
                f"(last appended was {self._records[-1].trial_id})"
            )

        self._records.append(record)

    def snapshot(self) -> tuple[TrialRecord, ...]:
        """Return a read-only snapshot of the current history."""
        return tuple(self._records)