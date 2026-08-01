"""
persistence.py

This file owns one optimization run’s on-disk layout. Its job is
keeping track of the run directory, handing out the correct per trial
paths, and writing history checkpoints. (Section 9 in the TDS)

KNOWN GAP!!! A few of the files from the spec aren’t implemented yet,
but that’s because the pieces they depend on don’t exist yet either.
pareto_front.csv, summary.txt, and the rest of the finalization outputs
all need an OptimizationResult, which in turn needs the ParetoFront
sweep that pareto.py intentionally doesn’t implement yet.

resolved_config.json is also missing for the same reason.
config_loader.py knows how to read JSON into project objects, but
nothing in the project knows how to serialize those objects back into
JSON at this point so that is for later

So for now RunDirectory only owns the parts that are actually
unblocked which is creating the run/per-trial directory
structure and writing the atomic history.csv checkpoints.
"""

from __future__ import annotations

import csv
import os
import secrets
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from black_box_optimizer.models import OptimizationContract
from black_box_optimizer.pareto import is_eligible
from black_box_optimizer.records import TrialRecord

_TRIAL_DIRECTORY_DIGITS = 4

# Fixed columns required by the TDS in order Any param.* columns and
# metric.* columns get inserted in the middle then error_message comes
# last
_FIXED_COLUMNS = (
    "trial_id",
    "execution_status",
    "metrics_status",
    "execution_succeeded",
    "objective_eligible",
    "runtime_seconds",
    "exit_code",
    "timed_out",
)


class RunDirectory:
    """Owns one run's directory layout, per-trial paths, and history checkpoints."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._trials_path = path / "trials"
        self._trials_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The run's own directory."""
        return self._path

    def trial_directory(self, trial_id: int) -> Path:
        """Create (if needed) and return one trial's own directory."""
        directory = self._trials_path / self._trial_directory_name(trial_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def metrics_path(self, trial_id: int) -> Path:
        """Where one trial's metrics.csv belongs; creates its directory first."""
        return self.trial_directory(trial_id) / "metrics.csv"

    def checkpoint(
        self,
        history: Sequence[TrialRecord],
        contract: OptimizationContract,
    ) -> None:
        """
        Atomically replace history.csv with the newest flattened history.

        The trick here is writing everything to a temporary file first,
        then swapping it into place with os.replace(). That way if
        something goes wrong halfway through, the previous checkpoint is
        still completely intact instead of being left half-written.

        If checkpointing fails, that's treated as a fatal error, per TDS
        section 9.4.
        """
        destination = self._path / "history.csv"
        fieldnames = _build_fieldnames(history, contract)

        fd, temp_name = tempfile.mkstemp(
            dir=self._path, prefix="history.", suffix=".csv.tmp"
        )
        try:
            with os.fdopen(fd, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in history:
                    writer.writerow(
                        _flatten_record(record, contract, fieldnames)
                    )
            os.replace(temp_name, destination)
        except OSError as error:
            Path(temp_name).unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to checkpoint history.csv; the previously "
                f"committed file (if any) was retained: {error}"
            ) from error

    @staticmethod
    def _trial_directory_name(trial_id: int) -> str:
        return f"trial_{trial_id:0{_TRIAL_DIRECTORY_DIGITS}d}"


def create_run_directory(base_directory: str | Path) -> RunDirectory:
    """
    Create a brand-new run directory.

    Runs are never resumed or overwritten here. Every optimization gets
    its own uniquely named directory so previous runs stay untouched.
    """
    base = Path(base_directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = secrets.token_hex(3)
    run_path = base / f"run_{timestamp}_{suffix}"
    run_path.mkdir(parents=True, exist_ok=False)
    return RunDirectory(run_path)


def _build_fieldnames(
    history: Sequence[TrialRecord], contract: OptimizationContract
) -> list[str]:
    """
    Build the history.csv column order.

    Start with the required fixed columns, then every declared
    parameter, then every metric we've seen so far (sorted just to keep
    the output deterministic), and finally error_message.
    """
    param_columns = [f"param.{p.name}" for p in contract.parameters]

    metric_names: set[str] = set()
    for record in history:
        metric_names.update(record.metrics.keys())
    metric_columns = [f"metric.{name}" for name in sorted(metric_names)]

    return [*_FIXED_COLUMNS, *param_columns, *metric_columns, "error_message"]


def _flatten_record(
    record: TrialRecord,
    contract: OptimizationContract,
    fieldnames: Sequence[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "trial_id": record.trial_id,
        "execution_status": record.execution_status,
        "metrics_status": record.metrics_status,
        "execution_succeeded": record.execution_succeeded,
        "objective_eligible": is_eligible(record, contract),
        "runtime_seconds": record.runtime_seconds,
        "exit_code": record.exit_code,
        "timed_out": record.timed_out,
        "error_message": record.error_message,
    }

    for parameter in contract.parameters:
        # Every declared parameter should exist in every TrialRecord
        # If one doesn't something went wrong earlier in the pipeline
        # and I think we all would rather fail loudly than quietly
        row[f"param.{parameter.name}"] = record.parameters[parameter.name]

    for field in fieldnames:
        if field.startswith("metric."):
            metric_name = field[len("metric.") :]
            row[field] = record.metrics.get(metric_name, "")

    return row