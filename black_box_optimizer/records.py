"""
records.py

This file defines TrialRecord, which stores everything we know about one
attempt to run the worker, and build_trial_record(), the factory function
that puts one TrialRecord together from a candidate's parameters and what
actually happened when the worker ran.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from black_box_optimizer.metrics import (
    MetricsFormatError,
    NonFiniteMetricError,
    read_trial_metrics,
)
from black_box_optimizer.models import CandidateConfiguration, ParameterValue

# These are the only execution and metrics states we expect to see
# Anything else means something upstream has gone wrong
ExecutionStatus = Literal[
    "completed", "process_failed", "timed_out", "launch_failed", "cancelled"
]
MetricsStatus = Literal["valid", "missing", "malformed", "nonfinite"]

# Keep these in sync with the literal definitions above
_VALID_EXECUTION_STATUSES = (
    "completed", "process_failed", "timed_out", "launch_failed", "cancelled"
)
_VALID_METRICS_STATUSES = ("valid", "missing", "malformed", "nonfinite")


def _is_number(value: object) -> bool:
    """Return True only for real ints and floats, not bools."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_plain_int(value: object) -> bool:
    """Return True only for real integers, not bools."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """
    Immutable evidence of one attempted worker execution.

    Once created, a TrialRecord should never change. It represents exactly
    what was tried and exactly what happened.
    """

    trial_id: int
    parameters: Mapping[str, ParameterValue]
    metrics: Mapping[str, float]
    execution_status: ExecutionStatus
    metrics_status: MetricsStatus
    runtime_seconds: float
    exit_code: int | None
    timed_out: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Validate every field and make the mappings read-only."""
        # Every trial needs a real non-negative int identifier
        if not _is_plain_int(self.trial_id):
            raise ValueError("trial_id must be an integer")
        if self.trial_id < 0:
            raise ValueError("trial_id cannot be negative")

        # Copy first so changes to the callers original dictionaries
        # can't sneaky change this record later
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )
        object.__setattr__(
            self, "metrics", MappingProxyType(dict(self.metrics))
        )

        # Only the status values defined above are valid
        if self.execution_status not in _VALID_EXECUTION_STATUSES:
            valid = sorted(_VALID_EXECUTION_STATUSES)
            raise ValueError(f"execution_status must be one of {valid}")
        if self.metrics_status not in _VALID_METRICS_STATUSES:
            valid = sorted(_VALID_METRICS_STATUSES)
            raise ValueError(f"metrics_status must be one of {valid}")

        # Runtime needs to be a real, finite, and non-negative number
        if not _is_number(self.runtime_seconds):
            raise ValueError("runtime_seconds must be numeric")
        if not math.isfinite(float(self.runtime_seconds)):
            raise ValueError("runtime_seconds must be finite")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds cannot be negative")

        # If the worker never actually exited there might not be an exit code
        if self.exit_code is not None and not _is_plain_int(self.exit_code):
            raise ValueError("exit_code must be an integer or None")
        if not isinstance(self.timed_out, bool):
            raise TypeError("timed_out must be a bool")
        if self.error_message is not None and not isinstance(
            self.error_message, str
        ):
            raise TypeError("error_message must be a string or None")

    @property
    def execution_succeeded(self) -> bool:
        """Return True only if the worker completed successfully."""
        return self.execution_status == "completed"


def build_trial_record(
    candidate: CandidateConfiguration,
    trial_id: int,
    metrics_path: str | Path,
    execution_result: Mapping[str, object],
) -> TrialRecord:
    """
    Build one immutable TrialRecord.

    execution_result is the dict runner.execute() returns: runtime_seconds,
    exit_code, timed_out, execution_status, and error_message.

    NOTE!!!!!! The design spec shows a shortened version of this signature. This
    implementation matches the real runner instead -- metrics_path is one of
    runner.execute()'s arguments, and trial_id isn't part of the runner's
    interface at all, it's assigned separately (probably by the controller).
    Confirm the runner's contract before changing this function's arguments. :)

    """
    metrics: Mapping[str, float] = {}
    metrics_status: MetricsStatus

    # Whether the worker ran successfully and whether it produced usable
    # metrics are independent so we record them separate.
    try:
        metrics = read_trial_metrics(metrics_path)
        metrics_status = "valid"
    except FileNotFoundError:
        # No metrics file was written
        metrics_status = "missing"
    except NonFiniteMetricError:
        # The file existed but at least one metric was NaN or infinite
        metrics_status = "nonfinite"
    except MetricsFormatError:
        # The file existed but didn't match the required CSV format
        metrics_status = "malformed"

    # Everything else comes directly from the candidate and whatever the
    # runner observed while executing the worker
    return TrialRecord(
        trial_id=trial_id,
        parameters=candidate.parameters,
        metrics=metrics,
        execution_status=execution_result["execution_status"],
        metrics_status=metrics_status,
        runtime_seconds=execution_result["runtime_seconds"],
        exit_code=execution_result["exit_code"],
        timed_out=execution_result["timed_out"],
        error_message=execution_result["error_message"],
    )